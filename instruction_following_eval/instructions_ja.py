# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library of instructions."""
import collections
import json
import random
import re
import string
from typing import Dict, Optional, Sequence, Union

from absl import logging
import langdetect

from instruction_following_eval import instructions_util

from janome.tokenizer import Tokenizer
import unicodedata


_InstructionArgsDtype = Optional[Dict[str, Union[int, str, Sequence[str]]]]

_LANGUAGES = instructions_util.LANGUAGE_CODES

_KANJI_NUM = 30

_ENDING_LETTERS = ("です", "ます")

_NOMINAL_ENDING_COUNT = 5

# The relational operation for comparison.
_COMPARISON_RELATION = ("未満", "以上")

# The maximum number of sentences.
_MAX_NUM_SENTENCES = 20

# The number of placeholders.
_NUM_PLACEHOLDERS = 4

# The number of bullet lists.
_NUM_BULLETS = 5

# The options of constrained response.
_CONSTRAINED_RESPONSE_OPTIONS = (
    "My answer is yes.", "My answer is no.", "My answer is maybe.")

# The options of starter keywords.
_STARTER_OPTIONS = ("I would say", "My answer is", "I believe",
                    "In my opinion", "I think", "I reckon", "I feel",
                    "From my perspective", "As I see it", "According to me",
                    "As far as I'm concerned", "To my understanding",
                    "In my view", "My take on it is", "As per my perception")

# The options of ending keywords.
# TODO(jeffreyzhou) add more ending options
_ENDING_OPTIONS = ("Any other questions?",
                   "Is there anything else I can help with?")

# The number of highlighted sections.
_NUM_HIGHLIGHTED_SECTIONS = 4

# The section spliter.
_SECTION_SPLITER = ("章", "節", "項")

# The number of sections.
_NUM_SECTIONS = 5

# The number of paragraphs.
_NUM_PARAGRAPHS = 5

# The postscript marker.
_POSTSCRIPT_MARKER = ("P.S.", "P.P.S", "追伸")

# The number of keywords.
_NUM_KEYWORDS = 2

# The occurrences of a single keyword.
_KEYWORD_FREQUENCY = 3

# The occurrences of a single letter.
_LETTER_FREQUENCY = 10

# The number of words in the response.
_NUM_LETTERS_LOWER_LIMIT = 200
_NUM_LETTERS_UPPER_LIMIT = 1000


class Instruction:
  """An instruction template."""

  def __init__(self, instruction_id):
    self.id = instruction_id

  def build_description(self, **kwargs):
    raise NotImplementedError("`build_description` not implemented.")

  def get_instruction_args(self):
    raise NotImplementedError("`get_instruction_args` not implemented.")

  def get_instruction_args_keys(self):
    raise NotImplementedError("`get_instruction_args_keys` not implemented.")

  def check_following(self, value):
    raise NotImplementedError("`check_following` not implemented.")


class KanjiLimit(Instruction):
  """回答中の漢字の使用回数を制限します。"""

  def build_description(self, *, kanji_limit=None, relation=None):
    """指示文を作成します。

    Args:
      kanji_limit: 漢字の使用回数を指定する整数。
      relation: 比較のための関係演算子を定義する文字列（`未満` または `以上`）。

    Returns:
      指示文を表す文字列。
    """
    self._kanji_limit = kanji_limit
    if self._kanji_limit is None or self._kanji_limit < 0:
      self._kanji_limit = random.randint(1, _KANJI_NUM)
    
    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError(f"サポートされている比較の関係は {_COMPARISON_RELATION} のいずれかである必要がありますが、{relation} が指定されました。")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
      "{kanji_limit}文字{relation}漢字を用いて、答えてください。") 
    return self._description_pattern.format(kanji_limit=self._kanji_limit, 
                                            relation=self._comparison_relation)
  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"kanji_limit": self._kanji_limit, "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["kanji_limit", "relation"]

  def check_following(self, value):
    """漢字の使用回数が指示に従っているか確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      漢字の使用回数が指示に従っていれば True、それ以外は False。
    """
    kanji_count = len(re.findall(r'[\u4e00-\u9faf]', value))
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return kanji_count < self._kanji_limit
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return kanji_count >= self._kanji_count


class NoHiragana(Instruction):
  """ひらがなを一つも使わずに回答しているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = ("ひらがなを一文字も使わないで答えてください。")
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return []

  def check_following(self, value):
    """ひらがなが使われていないか確認します。"""
    return not any('ぁ'<=char<='ゖ' for char in value)
  

class HiraganaOnly(Instruction):
  """ひらがなだけで回答しているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = ("ひらがなだけを用いて答えてください。")
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return []

  def check_following(self, value):
    """ひらがなだけで回答されているか確認します。"""
    def is_hiragana(char):
      return 'ぁ' <= char <= 'ん' or char == 'ー' or char == '・'

    def is_ignorable(char):
      return unicodedata.category(char).startswith('P') or unicodedata.category(char).startswith('S')

    return all(is_hiragana(char) or is_ignorable(char) for char in value)


class NoKatakana(Instruction):
  """カタカナを一つも使わずに回答しているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = ("カタカナを一文字も使わないで答えてください。")
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return []

  def check_following(self, value):
    """カタカナが使われていないか確認します。"""
    return not any(('ァ'<=char<='ヺ' or 'ｦ'<=char<='ﾝ') for char in value)


class KatakanaOnly(Instruction):
  """カタカナだけで回答しているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = ("カタカナだけを用いて答えてください。")
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return []

  def check_following(self, value):
    """カタカナだけで回答されているか確認します。"""
    def is_katakana(char):
      return ('ァ' <= char <= 'ン' or
              char == 'ー' or
              char == '・' or
              'ｦ' <= char <= 'ﾟ')

    def is_ignorable(char):
      return unicodedata.category(char).startswith('P') or unicodedata.category(char).startswith('S')

    return all(is_katakana(char) or is_ignorable(char) for char in value)


class SentenceEndingUnification(Instruction):
  """応答の中で各文の文末が統一されているか確認します。"""

  def build_description(self, *, ending=None):
    """指示文を作成します。
      
    Args:
      ending: 統一された全ての文末に使用される文字列。
    
    Returns:
      指示文を表す文字列。
    """
    self._ending = ending
    if self._ending is None:
        self._ending = random.choice(_ENDING_LETTERS)
    self._description_pattern = (
      "応答において、全ての文末が「{ending}」で統一された自然な文章にしてください。")
    return self._description_pattern.format(ending=self._ending)
  
  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"ending": self._ending}
  
  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["ending"]

  def check_following(self, value):
    """全ての文末が指定した形になっているか確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      文末が指示に従っている場合は True、それ以外は False。
    """
    quote_pattern_1 = re.compile(r'「.*?」')
    quote_pattern_2 = re.compile(r'『.*?』')
    value = re.sub(quote_pattern_1, '', value)
    value = re.sub(quote_pattern_2, '', value)

    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(value))

    sentences = re.split(r'[。！？]', value)
    for sentence in sentences:
      if sentence and not sentence.endswith(self._ending):
        return False
    return True
  

class NominalEndingChecker(Instruction):
  """体言止めが指定された回数以上使用されているか確認します。"""

  def build_description(self, *, count=None):
    """指示文を作成します。

    Args:
      count: 体言止めの使用回数を表す整数。

    Returns:
      指示文を表す文字列。
    """
    self._count = count 
    if self._count is None or self._count < 0:
      self._count = random.randint(1, _NOMINAL_ENDING_COUNT)
    self._description_pattern = ("応答の中で体言止めを{count}回は使用してください")
    return self._description_pattern.format(count = self._count)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"count": self._count}
  
  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["count"]
  
  def check_following(self, value):
    """体言止めが指示に従った回数使用されているか確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      体言止めの使用回数が指示に従っている場合は True、それ以外は False。
    """
    quote_pattern_1 = re.compile(r'「.*?」')
    quote_pattern_2 = re.compile(r'『.*?』')
    value = re.sub(quote_pattern_1, '', value)
    value = re.sub(quote_pattern_2, '', value)

    tokenizer = Tokenizer()
    tokens = list(tokenizer.tokenize(value))

    noun_count = 0
    for i in range(1, len(tokens)):
      if tokens[i].surface in '。！？' and tokens[i-1].part_of_speech.startswith('名詞'):
        noun_count += 1

    return noun_count >= self._count


class KanjiNumberNotationChecker(Instruction):
  """数字が全て漢数字で表記されているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = "数字を全て漢数字で表記する"
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return []
  
  def check_following(self, value):
    """数字が全て漢数字で表記されているか確認します。"""
    return not re.search(r'\d', value)


class ResponseLanguageChecker(Instruction):
  """応答の言語を確認します。"""

  def build_description(self, *, language = None):
    """指示文を作成します。

    Args:
      language: 応答に要求する言語を表す文字列。
        例えば、英語の場合は `en`、中国語の場合は `zh`、フランス語の場合は `fr` のように、
        ISO 639-1 コード (https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes) に従って記された
        `langid.py` (https://pypi.org/project/langid/1.1.5/) で定義された97種類のいずれかから言語は選んでください。

    Returns:
      指示文を表す文字列。
    """
    self._language = language
    if self._language is None:
      self._language = random.choice(list(_LANGUAGES.keys()))
    # TODO(tianjianlu): opens the description generation to more choices.
    self._description_pattern = (
        "あなたの応答全体は 言語「{language}」で記してください。他の言語は許可されません。")
    return self._description_pattern.format(language=_LANGUAGES[self._language])

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"language": self._language}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["language"]

  def check_following(self, value):
    """応答の全体の言語が指示に従っているか確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      `value` の言語が指示に従っている場合は True、それ以外の場合は False。
    """
    assert isinstance(value, str)

    try:
      return langdetect.detect(value) == self._language
    except langdetect.LangDetectException as e:
      # Count as instruction is followed.
      logging.error(
          "「%s」の言語を検出できませんでした。\n原因: %s ", value, e
      )  # refex: disable=pytotw.037
      return True


class NumberOfSentences(Instruction):
  """応答に含まれる文の数を確認します。"""

  def build_description(self, *, num_sentences = None,
                        relation = None):
    """指示文を作成します。

    Args:
      num_sentences: 閾値として指定する文の数を表す整数。
      relation: 比較のための関係演算子を定義する文字列（`未満` または `以上`）。

    Returns:
      指示文を表す文字列。
    """
    # The number of sentences as a threshold for comparison.
    self._num_sentences_threshold = num_sentences
    if (self._num_sentences_threshold is None or self._num_sentences_threshold < 0):
      self._num_sentences_threshold = random.randint(1, _MAX_NUM_SENTENCES)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError(f"サポートされている比較の関係は {_COMPARISON_RELATION} のいずれかである必要がありますが、{relation} が指定されました。")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "応答は{num_sentences}文{relation}の文章で構成させてください。")
    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_sentences=self._num_sentences_threshold)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_sentences": self._num_sentences_threshold,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_sentences", "relation"]

  def check_following(self, value):
    """文の数が指示に従っているかを確認します。

    Args:
      value: 回答を表す文字列。

    Returns:
      回答が指示に従っている場合は True。
    """
    num_sentences = instructions_util.count_sentences(value)
    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_sentences < self._num_sentences_threshold
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_sentences >= self._num_sentences_threshold


class PlaceholderChecker(Instruction):
  """テンプレートを記述する際に適切な数のプレースホルダーを含められるか確認します。"""

  def build_description(self, *, num_placeholders = None):
    """指示文を作成します。

    Args:
      num_placeholders: 応答に必要なプレースホルダーの最小数を示す整数。

    Returns:
      指示文を表す文字列。
    """
    self._num_placeholders = num_placeholders
    if self._num_placeholders is None or self._num_placeholders < 0:
      self._num_placeholders = random.randint(1, _NUM_PLACEHOLDERS)
    self._description_pattern = (
        "応答には少なくとも {num_placeholders} 個のプレースホルダーが含まれている必要があります。" +
        "プレースホルダーは [名前] のように角括弧で表されます。")
    return self._description_pattern.format(
        num_placeholders=self._num_placeholders)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_placeholders": self._num_placeholders}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_placeholders"]

  def check_following(self, value):
    """プレースホルダーの数が指示に従っているか確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      実際のプレースホルダーの数が `num_placeholders` 以上であれば True、それ以外は False。
    """
    placeholders = re.findall(r"\[.*?\]", value)
    num_placeholders = len(placeholders)
    return num_placeholders >= self._num_placeholders


class BulletListChecker(Instruction):
  """応答が箇条書きで書かれているか確認します。"""

  def build_description(self, *, num_bullets = None):
    """指示文を作成します。

    Args:
      num_bullets: 応答に含めるべき箇条書きの数を指定する整数。

    Returns:
      指示文を表す文字列。
    """
    self._num_bullets = num_bullets
    if self._num_bullets is None or self._num_bullets < 0:
      self._num_bullets = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "応答はちょうど {num_bullets} 個の箇条書きで構成してください。 " +
        "以下のような箇条書きの形を参考にしてください:\n" +
        "・一つめの内容\n" +
        "・二つめの内容")
    return self._description_pattern.format(
        num_bullets=self._num_bullets)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_bullets": self._num_bullets}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_bullets"]

  def check_following(self, value):
    r"""箇条書きの数が要件を満たしているかをチェックします。

    Args:
      value: 応答を表す文字列。応答には `・` で始まる箇条書きが含まれていることが期待されます。

    Returns:
      実際の箇条書きの数が要件を満たしている場合は True を返します。
    """
    bullet_lists = re.findall(r"^\s*・[^\・].*$", value, flags=re.MULTILINE)
    num_bullet_lists = len(bullet_lists)
    return num_bullet_lists == self._num_bullets


class NumberedListChecker(Instruction):
  """応答が数字のリストで書かれているか確認します。"""

  def build_description(self, *, num_items = None):
    """指示文を作成します。

    引数:
      num_items: 応答に含めるべき番号付きリストの数を指定する整数。

    戻り値:
      指示文を表す文字列。
    """
    self._num_items = num_items
    if self._num_items is None or self._num_items < 0:
      self._num_items = random.randint(1, _NUM_BULLETS)
    self._description_pattern = (
        "応答はちょうど {num_bullets} 個の番号付きリストで構成してください。 " +
        "以下のような番号付きリストの形を参考にしてください:\n" +
        "1. 一つめの内容\n" +
        "2. 二つめの内容")
    return self._description_pattern.format(
        num_items=self._num_items)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_items": self._num_items}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_items"]

  def check_following(self, value):
    r"""番号付きリストの数が要件を満たしているかをチェックします。

    引数:
      value: 応答を表す文字列。応答には `1. ` で始まる番号付きリストが含まれていることが期待されます。

    戻り値:
      実際の番号付きリストの数が要件を満たしている場合は True を返します。
    """
    numbered_lists = re.findall(r"^\s*\d+\.\s.*$", value, flags=re.MULTILINE)
    num_numbered_lists = len(numbered_lists)
    return num_numbered_lists == self._num_items
  

class ConstrainedResponseChecker(Instruction):
  """指定された応答を返すか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._constrained_responses = _CONSTRAINED_RESPONSE_OPTIONS
    self._description_pattern = ("次の選択肢のいずれかで回答してください: {response_options}")
    return self._description_pattern.format(
        response_options=self._constrained_responses)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """指定された選択肢の中から応答が選ばれたかを確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      実際の応答が指定された選択肢のいずれかを含む場合はTrue、それ以外の場合はFalseを返します。
    """
    value = value.strip()
    for constrained_response in self._constrained_responses:
      if constrained_response in value:
        return True
    return False


class ConstrainedStartChecker(Instruction):
  """応答の書き始めを確認します。"""

  def build_description(self, *, starter = None):
    """指示文を作成します。

    Args:
      starter: 応答の書き始めを示すキーワードを表す文字列。

    Returns:
      指示文を表す文字列。
    """
    self._starter = starter.strip() if isinstance(starter, str) else starter
    if self._starter is None:
      self._starter = random.choice(_STARTER_OPTIONS)
    self._description_pattern = ("会話中あなたの番になったら、必ず{starter}で応答を始めてください。")
    return self._description_pattern.format(starter=self._starter)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"starter": self._starter}

  def get_instruction_args_keys(self):
    """`build_description`の引数のキーを返します。"""
    return ["starter"]

  def check_following(self, value):
    """応答が指定されたキーワードやフレーズで始まっているかをチェックします。

    Args:
      value: 応答を表す文字列。

    Returns:
      応答が`instruction_args`に含まれるフレーズやキーワードで始まっている場合はTrue、
      そうでない場合はFalse。
    """
    response_pattern = r"^\s*" + self._starter + r".*$"
    response_with_constrained_start = re.search(response_pattern, value,
                                                flags=re.MULTILINE)
    return True if response_with_constrained_start else False


class HighlightSectionChecker(Instruction):
  """ハイライトされたセクションの数をチェックします。"""

  def build_description(self, *, num_highlights = None):
    """指示文を作成します。

    Args:
      num_highlights: ハイライトされたセクションの数を指定する整数。

    Returns:
      指示文を表す文字列。
    """
    self._num_highlights = num_highlights
    if self._num_highlights is None or self._num_highlights < 0:
      self._num_highlights = random.randint(1, _NUM_HIGHLIGHTED_SECTIONS)

    self._description_pattern = (
        "例えば《強調されたセクション》のように、回答の中で少なくとも{num_highlights}つのセクションを《》の記号を用いて強調してください。")

    return self._description_pattern.format(num_highlights=self._num_highlights)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"num_highlights": self._num_highlights}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["num_highlights"]

  def check_following(self, value):
    """指定された数以上のセクションが強調されているか確認します。

    Args:
      value: 回答を表す文字列。回答は《》で囲まれた強調されたセクションを有していることが期待されます。

    Returns:
      回答内の強調されたセクションの数が最小セクション数以上であればTrue、それ以外はFalse。
    """
    num_highlights = 0
    highlights = re.findall(r"《[^\n《》]*》", value)
    for highlight in highlights:
      if highlight.strip("《》").strip():
        num_highlights += 1

    return num_highlights >= self._num_highlights


class SectionChecker(Instruction):
  """セクションの数をチェックします。"""

  def build_description(self, *, section_spliter = None,
                        num_sections = None):
    """指示文を作成します。

    Args:
      section_spliter: 「章」や「節」などの新しいセクションの始まりを示すキーワードを表す文字列。
      num_sections: セクションの数を指定する整数。

    Returns:
      指示文を表す文字列。
    """
    self._section_spliter = section_spliter.strip() if isinstance(
        section_spliter, str) else section_spliter
    if self._section_spliter is None:
      self._section_spliter = random.choice(_SECTION_SPLITER)

    self._num_sections = num_sections
    if self._num_sections is None or self._num_sections < 0:
      self._num_sections = random.randint(1, _NUM_SECTIONS)

    self._description_pattern = (
        "あなたの回答は{num_sections}つのセクションで構成されている必要があります。" +
        "各セクションの始まりは数字と{section_spliter}から書き始めてください。例:\n" +
        "第1{section_spliter}\n" +
        "[セクション1の内容]\n" +
        "第2{section_spliter}\n" +
        "[セクション2の内容]")

    return self._description_pattern.format(
        num_sections=self._num_sections,
        section_spliter=self._section_spliter)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"section_spliter": self._section_spliter,
            "num_sections": self._num_sections}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["section_spliter", "num_sections"]

  def check_following(self, value):
    """回答が複数のセクションで構成されているか確認します。

    Args:
      value: 回答を表す文字列。回答は1より大きい複数のセクションで構成されていることが期待されます。
        新しいセクションは`第１章`のようにセクション番号で始まります。

    Returns:
      回答のセクション数が最小セクション数以上であればTrue、それ以外はFalse。
    """
    section_splitter_patten = r"第[\d\uFF10-\uFF19]+章"
    sections = re.split(section_splitter_patten, value)
    num_sections = len(sections) - 1
    return num_sections >= self._num_sections


class ParagraphChecker(Instruction):
  """段落の数を確認します。"""

  def build_description(self, *, num_paragraphs = None):
    """指示文を作成します。

    Args:
      num_paragraphs: 段落の数を指定する整数。

    Returns:
      指示文を表す文字列。
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._description_pattern = (
        "応答は{num_paragraphs}個の段落に分かれた文章で送ってください。それぞれの段落をマークダウンの区切り記号: *** で区切ってください。")

    return self._description_pattern.format(num_paragraphs=self._num_paragraphs)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"num_paragraphs": self._num_paragraphs}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["num_paragraphs"]

  def check_following(self, value):
    """応答が必要な段落数を含んでいるかを確認します。

    Args:
      value: 応答を表す文字列。応答にはマークダウンの区切り記号: `***` で区切られた段落が含まれていることが期待されます。


    Returns:
      実際の段落数が要求された数と同じ場合はTrue、それ以外の場合はFalse。
    """
    paragraphs = re.split(r"\s?\*\*\*\s?", value)
    num_paragraphs = len(paragraphs)

    for index, paragraph in enumerate(paragraphs):
      if not paragraph.strip():
        if index == 0 or index == len(paragraphs) - 1:
          num_paragraphs -= 1
        else:
          return False

    return num_paragraphs == self._num_paragraphs


class PostscriptChecker(Instruction):
  """追伸が含まれているか確認します。"""

  def build_description(self, *, postscript_marker = None):
    """指示文を作成します。

    Args:
      postscript_marker: 追伸の開始を示すキーワードを含む文字列。

    Returns:
      指示文を表す文字列。
    """
    self._postscript_marker = postscript_marker.strip() if isinstance(postscript_marker, str) else postscript_marker
    if self._postscript_marker is None:
      self._postscript_marker = random.choice(_POSTSCRIPT_MARKER)

    self._description_pattern = ("応答の最後に、{postscript}で始まる追伸を追加してください。")

    return self._description_pattern.format(postscript=self._postscript_marker)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"postscript_marker": self._postscript_marker}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["postscript_marker"]

  def check_following(self, value):
    """応答が追伸形式に従っているかを確認します。

    Args:
      value: 応答を表す文字列。応答には追伸セクションが含まれていることが期待されます。

    Returns:
      応答が`instruction_args`に含まれるキーワードで始まる追伸セクションを含んでいる場合はTrue、それ以外の場合はFalse。
    """
    if self._postscript_marker == "P.P.S":
      postscript_pattern = r"\s*p\.\s?p\.\s?s.*$"
    elif self._postscript_marker == "P.S.":
      postscript_pattern = r"\s*p\.\s?s\..*$"
    else:
      postscript_pattern = r"\s*" + re.escape(self._postscript_marker) + r".*$"
    postscript = re.findall(postscript_pattern, value, flags=re.IGNORECASE | re.MULTILINE)
    return True if postscript else False


class RephraseChecker(Instruction):
  """指定した箇所を言い換えしているかを確認します。"""

  def build_description(self, *, original_message):
    """指示文を作成します。

    Args:
      original_message: 元のメッセージを表す文字列。言い換えた応答は、波括弧で囲まれた部分のみを変更させます。
        例えば、{ここを変更} のように。元のメッセージと再構成されたメッセージの両方に、{ここを変更} の形式で変更が含まれている必要があります。

    Returns:
      指示文を表す文字列。
    """
    if not self.is_change(original_message):
      raise ValueError(f"メッセージ {original_message} には {{ここを変更}} の形式で変更が含まれていません。")


    self._reference_without_change = original_message
    self._description = ("例えば {ここを変更} のように、元の文章を波括弧で囲まれた部分のみを変更して応答してください")
    return self._description

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"original_message": self._reference_without_change}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["original_message"]

  def check_following(self, value):
    r"""指示に従って言い換えられているか確認します。

    Args:
      value: 応答を表す文字列で、`instruction_args`の文字列を言い換えたものが期待されます。

    Returns:
      `value`と`instruction_args`が波括弧で囲まれた部分のみ異なる場合はTrue、それ以外の場合はFalseを返します。
    """

    if not self.is_change(value):
      raise ValueError(f"value {value} には {{ここを変更}} の形式で変更が含まれていません。")


    response_without_changes = self.strip_changes(value)
    reference_without_changes = self.strip_changes(
        self._reference_without_change)

    return response_without_changes == reference_without_changes

  def is_change(self, response):
    """応答に {ここを変更} の形式で変更が含まれているか確認します。"""
    return re.search(r"\{.*\}", response)

  def strip_changes(self, response):
    """変更部分を取り除きます。"""
    return re.sub(r"\{.*\}", "", response)


class KeywordChecker(Instruction):
  """キーワードが含まれているか確認します。"""

  def build_description(self, *, keywords = None):
    """指示文を作成します。

    Args:
      keywords: 応答に含まれるべきキーワードの文字列のシーケンス。

    Returns:
      指示文を表す文字列。
    """

    if not keywords:
      self._keywords = instructions_util.generate_keywords(num_keywords=_NUM_KEYWORDS)
    else:
      self._keywords = keywords
    self._keywords = sorted(self._keywords)

    self._description_pattern = ("応答に次のキーワード {keywords} を含めてください。")

    return self._description_pattern.format(keywords=self._keywords)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"keywords": self._keywords}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["keywords"]

  def check_following(self, value):
    """応答に要求したキーワードが含まれているか確認します。"""
    for keyword in self._keywords:
      if not re.search(keyword, value):
        return False
    return True


class KeywordFrequencyChecker(Instruction):
  """キーワードの出現頻度を確認します。"""

  def build_description(self, *, keyword = None,
                        frequency = None,
                        relation = None):
    """指示文を作成します。

    Args:
      keyword: 応答に含むよう要求したキーワードを表す文字列。
      frequency: `keyword` が応答全体に出現することが期待される回数を指定する整数。
      relation: 比較のための関係演算子を定義する文字列 (`未満`, `以上`)。

    Returns:
      指示文を表す文字列。
    """
    if not keyword:
      self._keyword = instructions_util.generate_keywords(num_keywords=1)[0]
    else:
      self._keyword = keyword.strip()

    self._frequency = frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _KEYWORD_FREQUENCY)

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("サポートされている比較のための関係演算子は "
                       f"{_COMPARISON_RELATION} のいずれかでなければなりませんが、{relation} が指定されました。")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "応答の中で、{keyword} という単語を{frequency}回{relation}出現させてください。")

    return self._description_pattern.format(
        keyword=self._keyword,
        relation=self._comparison_relation,
        frequency=self._frequency)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"keyword": self._keyword,
            "frequency": self._frequency,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["keyword", "frequency", "relation"]

  def check_following(self, value):
    """応答が指定した回数キーワードを含んでいるか確認します。"""
    actual_occurrences = len(re.findall(self._keyword, value))

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return actual_occurrences < self._frequency
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return actual_occurrences >= self._frequency


class NumberOfLetters(Instruction):
  """文字数を確認します。"""

  def build_description(self, *, num_letters=None, relation=None):
    """指示文を作成する。

    Args:
      num_letters: 応答に含まれる文字数を指定する整数。
      relation: 比較のための関係演算子を定義する文字列（`未満` または `以上`）。
        現在サポートされている関係演算子は2つ:
        '未満' の場合、実際の文字数 < num_letters;
        '以上' の場合、実際の文字数 >= num_letters。

    Returns:
      指示文を表す文字列。
    """

    self._num_letters = num_letters
    if self._num_letters is None or self._num_letters < 0:
      self._num_letters = random.randint(
          _NUM_LETTERS_LOWER_LIMIT, _NUM_LETTERS_UPPER_LIMIT
      )

    if relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif relation not in _COMPARISON_RELATION:
      raise ValueError("サポートされている比較の関係は "
                       f"{_COMPARISON_RELATION} のいずれかでなければなりませんが、{relation} が指定されました。")
    else:
      self._comparison_relation = relation

    self._description_pattern = (
        "{num_letters}文字{relation}で答えてください。")

    return self._description_pattern.format(
        relation=self._comparison_relation,
        num_letters=self._num_letters)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_letters": self._num_letters,
            "relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_letters", "relation"]

  def check_following(self, value):
    """応答が指定した文字数を含んでいるかをチェックします。"""
    num_letters = len(value)

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return num_letters < self._num_letters
    elif self._comparison_relation == _COMPARISON_RELATION[1]:
      return num_letters >= self._num_letters


class JsonFormat(Instruction):
  """JSON形式かどうかを確認します。"""

  def build_description(self):
    self._description_pattern = (
        "出力全体をJSON形式で囲んでください。マークダウンのバッククォート（```）などを使用してください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    value = (
        value.strip()
        .removeprefix("```json")
        .removeprefix("```Json")
        .removeprefix("```JSON")
        .removeprefix("```")
        .removesuffix("```")
        .strip()
    )
    try:
      json.loads(value)
    except ValueError as _:
      return False
    return True


class ParagraphFirstWordCheck(Instruction):
  """段落の数とn番目の段落の最初の単語を確認します。"""

  def build_description(self, num_paragraphs = None,
                        nth_paragraph = None,
                        first_word = None):
    r"""指示文を作成します。

    Args:
      num_paragraphs: 応答に要求する段落の数を示す整数。段落は文字列の一部であり、
        '\n\n' で区切られるよう要求します。
      nth_paragraph: 注目すべき段落の番号を示す整数。nは1から始まります。
      first_word: n番目の段落の最初の単語を表す文字列。

    Returns:
      指示文を表す文字列。
    """
    self._num_paragraphs = num_paragraphs
    if self._num_paragraphs is None or self._num_paragraphs < 0:
      self._num_paragraphs = random.randint(1, _NUM_PARAGRAPHS)

    self._nth_paragraph = nth_paragraph
    if (
        self._nth_paragraph is None
        or self._nth_paragraph <= 0
        or self._nth_paragraph > self._num_paragraphs
    ):
      self._nth_paragraph = random.randint(1, self._num_paragraphs + 1)

    self._first_word = first_word
    if self._first_word is None:
      self._first_word = instructions_util.generate_keywords(num_keywords=1)[0]

    self._description_pattern = (
        "{num_paragraphs}個の段落に分けて応答を書いてください。 " +
        "Pythonだと'\\n\\n'で表されるように、段落はそれぞれ2つの改行で区切ってください。 " +
        "{nth_paragraph}段落目は「{first_word} 」という単語で書き始めてください。")

    return self._description_pattern.format(
        num_paragraphs=self._num_paragraphs,
        nth_paragraph=self._nth_paragraph,
        first_word=self._first_word)

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"num_paragraphs": self._num_paragraphs,
            "nth_paragraph": self._nth_paragraph,
            "first_word": self._first_word}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["num_paragraphs", "nth_paragraph", "first_word"]

  def check_following(self, value):
    """総段落数と指定した箇所の書き初めの単語が正しいかを確認します。

    Args:
      value: 応答の内容を表す文字列。応答は段落で構成されており、それぞれ2つの改行で区切られている必要があります。
        n番目の段落の最初の単語は指定された単語と一致する必要があります。

    Returns:
      段落の数が要求通りであり、指定された段落の最初の単語が要求通りであればTrue。それ以外はFalse。
    """

    paragraphs = re.split(r"\n\n", value)
    num_paragraphs = len(paragraphs)

    for paragraph in paragraphs:
      if not paragraph.strip():
        num_paragraphs -= 1

    if self._nth_paragraph <= num_paragraphs:
      paragraph = paragraphs[self._nth_paragraph - 1].strip()
      if not paragraph:
        return False
    else:
      return False

    first_word = ""

    paragraph = paragraph.lstrip("「")

    first_word = paragraph[:len(self._first_word)]

    return (
        num_paragraphs == self._num_paragraphs
        and first_word == self._first_word
    )


# # TODO(jeffrey) add relation - at least/at most?
# class KeySentenceChecker(Instruction):
#   """Check the existence of certain key sentences."""

#   def build_description(self, key_sentences = None,
#                         num_sentences = None):
#     """Build the instruction description.

#     Args:
#       key_sentences: A sequences of strings representing the key sentences that
#         are expected in the response.
#       num_sentences: The number of key sentences that are expected to be seen in
#         the response.

#     Returns:
#       A string representing the instruction description.
#     """

#     if not key_sentences:
#       # TODO(jeffrey) make a generate sentences function? wonderwords package
#       self._key_sentences = set(["For now, this is fine."])
#     else:
#       self._key_sentences = key_sentences

#     if not num_sentences:
#       self._num_sentences = random.randint(1, len(self._key_sentences))
#     else:
#       self._num_sentences = num_sentences

#     self._description_pattern = (
#         "Include {num_sentences} of the following sentences {key_sentences}"
#     )

#     return self._description_pattern.format(
#         num_sentences=self._num_sentences, key_sentences=self._key_sentences
#     )

#   def get_instruction_args(self):
#     """Returns the keyward args of `build_description`."""
#     return {"num_sentences": self._num_sentences,
#             "key_sentences": list(self._key_sentences)}

#   def get_instruction_args_keys(self):
#     """Returns the args keys of `build_description`."""
#     return ["num_sentences", "key_sentences"]

#   def check_following(self, value):
#     """Checks if the response contains the expected key sentences."""
#     count = 0
#     sentences = instructions_util.split_into_sentences(value)
#     for sentence in self._key_sentences:
#       if sentence in sentences:
#         count += 1

#     return count == self._num_sentences


class ForbiddenWords(Instruction):
  """指定された単語が応答に含まれていないことを確認します。"""

  def build_description(self, forbidden_words = None
                        ):
    """指示文を作成します。

    Args:
      forbidden_words: 応答に含めてはいけない単語のリスト。

    Returns:
      指示文を表す文字列。
    """

    if not forbidden_words:
      self._forbidden_words = instructions_util.generate_keywords(
          num_keywords=_NUM_KEYWORDS)
    else:
      self._forbidden_words = list(set(forbidden_words))
    self._forbidden_words = sorted(self._forbidden_words)
    self._description_pattern = (
        "応答に {forbidden_words} という単語を含めないでください。"
    )

    return self._description_pattern.format(
        forbidden_words=self._forbidden_words
    )

  def get_instruction_args(self):
    """`build_description` のキーワード引数を返します。"""
    return {"forbidden_words": self._forbidden_words}

  def get_instruction_args_keys(self):
    """`build_description` の引数キーを返します。"""
    return ["forbidden_words"]

  def check_following(self, value):
    """応答に指定されたキーワードが含まれていないかを確認します。"""
    for word in self._forbidden_words:
      if re.search(word, value):
        return False
    return True


class RephraseParagraph(Instruction):
  """文章が言い換えられていることを確認するクラス。"""

  def build_description(self, *, original_paragraph, low, high):
    """指示文を作成する。

    Args:
      original_paragraph: 元の文章を表す文字列。
      low: 含めるべき同じ単語の最小数を表す整数。
      high: 含めるべき同じ単語の最大数を表す整数。

    Returns:
      指示文を表す文字列。
    """
    self._original_paragraph = original_paragraph
    self._low = low
    self._high = high

    self._description = (
      "次の文章を言い換えてください: " +
      "{original_paragraph}\nあなたの回答には、" +
      "元の文章に含まれている単語を{low}個から{high}個含める必要があります。" +
      "単語が同じであるとみなされるのは、すべての文字が同じ場合のみです。" +
      "例えば、'あめ'と'アメ'と'雨'は異なります。"
    )

    return self._description.format(original_paragraph=original_paragraph,
                                    low=self._low, high=self._high)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"original_paragraph": self._original_paragraph,
            "low": self._low,
            "high": self._high}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["original_paragraph", "low", "high"]

  def check_following(self, value):
    tokenizer = Tokenizer()
    val_words = [token.surface for token in tokenizer.tokenize(value) if not (token.part_of_speech.startswith('助詞') or token.part_of_speech.startswith('助動詞') or token.part_of_speech.startswith('記号'))]
    original_words = [token.surface for token in tokenizer.tokenize(self._original_paragraph) if not (token.part_of_speech.startswith('助詞') or token.part_of_speech.startswith('助動詞') or token.part_of_speech.startswith('記号'))]

    dict_val = collections.Counter(val_words)
    dict_original = collections.Counter(original_words)

    similar_words = 0
    for word in dict_original:
      similar_words += min(dict_original[word], dict_val[word])

    return similar_words >= self._low and similar_words <= self._high

  
class TwoResponsesChecker(Instruction):
  """2種類の応答があることを確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "2種類の異なる応答をしてください。それぞれの応答は全角のダッシュ記号6つ（ーーーーーー）で区切ってください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """応答が2つの異なる答えを含んでいるかを確認します。

    Args:
      value: 応答を表す文字列。

    Returns:
      2つの応答が検出された場合はTrue、それ以外の場合はFalse。
    """
    valid_responses = list()
    responses = value.split("ーーーーーー")
    for index, response in enumerate(responses):
      if not response.strip():
        if index != 0 and index != len(responses) - 1:
          return False
      else:
        valid_responses.append(response)
    return (
        len(valid_responses) == 2
        and valid_responses[0].strip() != valid_responses[1].strip()
    )


class RepeatPromptThenAnswer(Instruction):
  """最初にリクエストを繰り返し、その後に答えることを確認します。"""

  def build_description(self, *, prompt_to_repeat = None):
    """指示文を作成します。

    Args:
      prompt_to_repeat: 繰り返すべきプロンプト。

    Returns:
      指示文を表す文字列。
    """
    if not prompt_to_repeat:
      raise ValueError("prompt_to_repeatを設定する必要があります。")
    else:
      self._prompt_to_repeat = prompt_to_repeat
    self._description_pattern = (
        "最初にリクエストを一言一句変えずに繰り返し、その後に答えを述べてください"
        "（1. 繰り返す前に言葉や文字を追加しないこと; 2. 繰り返すべきリクエストにはこの文を含めないこと）"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"prompt_to_repeat": self._prompt_to_repeat}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["prompt_to_repeat"]

  def check_following(self, value):
    if value.strip().startswith(self._prompt_to_repeat.strip()):
      return True
    return False


class EndChecker(Instruction):
  """応答が指定されたフレーズで終わることをチェックします。"""

  def build_description(self, *, end_phrase = None):
    """指示文を作成します。

    Args:
      end_phrase: 応答が最後に出力すべきフレーズを表す文字列

    Returns:
      指示文を表す文字列。
    """
    self._end_phrase = (
        end_phrase.strip() if isinstance(end_phrase, str) else end_phrase
    )
    if self._end_phrase is None:
      self._end_phrase = random.choice(_ENDING_OPTIONS)
    self._description_pattern = (
        "応答の最後に次のフレーズをそのまま出力してください: {ender}。"
        "このフレーズの後に他の言葉を続けてはいけません。")
    return self._description_pattern.format(ender=self._end_phrase)

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"end_phrase": self._end_phrase}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return ["end_phrase"]

  def check_following(self, value):
    """応答が指定したフレーズで終わっているかをチェックします。"""
    value = value.strip().strip("\"")
    self._end_phrase = self._end_phrase.strip()
    return value.endswith(self._end_phrase)


class TitleChecker(Instruction):
  """応答にタイトルが含まれているかをチェックします。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "例えば『喜びの詩』のように、応答に二重山括弧で囲まれたタイトルをつけてください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """応答にタイトルが含まれているかをチェックします。"""
    pattern = r"『[^』]+』"
    re_pattern = re.compile(pattern)
    titles = re.findall(re_pattern, value)

    for title in titles:
      if title.lstrip("『").rstrip("』").strip():
        return True
    return False


class LetterFrequencyChecker:
  """文字の出現頻度をチェックします"""

  def build_description(self, *, letter=None, let_frequency=None, let_relation=None):
    """指示文を作成します。

    Args:
      letter: 出現回数を数える文字を表す文字列。
      let_frequency: `letter`が回答に出現する回数を指定する整数。
      let_relation: 比較のための関係演算子を定義する文字列（`未満`, `以上`）。

    Returns:
      指示文を表す文字列。
    """
    if not letter or len(letter) > 1 or not ('ぁ' <= letter <= 'ん'):
      self._letter = random.choice([chr(i) for i in range(ord('ぁ'), ord('ん') + 1)])
    else:
      self._letter = letter.strip()
      
    self._frequency = let_frequency
    if self._frequency is None or self._frequency < 0:
      self._frequency = random.randint(1, _LETTER_FREQUENCY)

    if let_relation is None:
      self._comparison_relation = random.choice(_COMPARISON_RELATION)
    elif let_relation not in _COMPARISON_RELATION:
      raise ValueError(
          "サポートされている比較の関係は "
          f"{_COMPARISON_RELATION} のいずれかでなければなりませんが、{let_relation} が指定されました。"
      )
    else:
      self._comparison_relation = let_relation

    self._description_pattern = (
        "応答には、文字「{letter}」を{let_frequency}回{let_relation}出現させてください。"
    )

    return self._description_pattern.format(
      letter=self._letter,
      let_frequency=self._frequency,
      let_relation=self._comparison_relation,
    )

  def get_instruction_args(self):
    """`build_description`のキーワード引数を返します。"""
    return {"letter": self._letter,
            "let_frequency": self._frequency,
            "let_relation": self._comparison_relation}

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返す。"""
    return ["letter", "let_frequency", "let_relation"]

  def check_following(self, value):
    """応答に指定された頻度で文字が含まれているかをチェックします。"""
    letters = collections.Counter(value)

    katakana_letter = chr(ord(self._letter) + 96)

    total_count = letters[self._letter] + letters[katakana_letter]

    if self._comparison_relation == _COMPARISON_RELATION[0]:
      return total_count < self._frequency
    else:
      return total_count >= self._frequency


class FuriganaForKanji(Instruction):
  """全ての漢字にふりがながついているか確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "全ての漢字にふりがなをつけてください。ふりがなは（）の中に書いてください。"
    )
    return self._description_pattern
  
  def get_instruction_args(self):
    """build_descriptionのキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """全ての漢字にふりがながついているか確認します。"""
    kanji_pattern = r'[\u4e00-\u9faf]+'
    kanji_with_furigana_pattern = r'[\u4e00-\u9faf]+（[ぁ-ん]+）'

    kanji_count = len(re.findall(kanji_pattern, value))
    kanji_with_furigana_count = len(re.findall(kanji_with_furigana_pattern, value))

    return kanji_count == kanji_with_furigana_count



class PeriodChecker(Instruction):
  """応答に句点が含まれていないかを確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "応答全体で句点を使用しないでください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """build_descriptionのキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """応答に句点が含まれていないかを確認します。"""
    return not re.search(r"\。", value)


class CommaChecker(Instruction):
  """応答に読点が含まれていないかを確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "応答全体で読点を使用しないでください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """build_descriptionのキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """応答に読点が含まれていないかを確認します。"""
    return not re.search(r"\、", value)


class QuotationChecker(Instruction):
  """応答が鉤括弧で囲まれているかを確認します。"""

  def build_description(self):
    """指示文を作成します。"""
    self._description_pattern = (
        "応答全体を鉤括弧で囲んでください。"
    )
    return self._description_pattern

  def get_instruction_args(self):
    """build_descriptionのキーワード引数を返します。"""
    return None

  def get_instruction_args_keys(self):
    """`build_description`の引数キーを返します。"""
    return []

  def check_following(self, value):
    """応答が鉤括弧で囲まれているかを確認します。"""
    value = value.strip()
    return len(value) > 1 and value[0] == '「' and value[-1] == '」'
