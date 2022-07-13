// Copyright 2022 The Google Research Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * GoogDOM is simplified and short derivative of goog.dom, used for
 *   convenience methods for creating HTML elements.
 */
class GoogDOM {
  constructor() {
    /** @private @const {!RegExp} */
    this.UNSAFE_RE_ = /<(?:math|script|style|svg|template)[^>]*>/i;
  }

  /**
   * Sets innerHTML of an element, throwing an exception if the passed markup
   *   seems unsafe.
   * @param {!Element} elt
   * @param {string} html
   */
  setInnerHtml(elt, html) {
    if (this.UNSAFE_RE_.test(html)) {
      throw 'SetInnerHtml found unsafe HTML';
    }
    elt.innerHTML = html;
  }

  /**
   * Creates a DOM element, accepting a variable number of args.
   *   The first argument should be the tag name, the second should be
   *   a class name or an array of class names or a dict of attributes.
   *   Any subsequent args are child elements.
   * @return {!Element}
   */
  createDom() {
    const tagName = String(arguments[0]);
    const attributes = arguments[1];
    const element = document.createElement(tagName);
    if (attributes) {
      if (typeof attributes === 'string') {
        element.className = attributes;
      } else if (Array.isArray(attributes)) {
        element.className = attributes.join(' ');
      } else {
        for (let k in attributes) {
          if (k == 'class') {
            element.className = attributes[k];
          } else {
            element.setAttribute(k, attributes[k]);
          }
        }
      }
    }
    if (arguments.length > 2) {
      this.append_(element, arguments, 2);
    }
    return element;
  }

  /**
   * Helper method for createDom(): appends each args[i] as a child to parent,
   *   starting at i = startIndex. args can include HTML markup as well as
   *   Elements.
   * @param {!Element} parent
   * @param {!Array} args
   * @param {number} startIndex
   * @return {!Element}
   */
  append_(parent, args, startIndex) {
    for (let i = startIndex; i < args.length; i++) {
      const child = args[i];
      if (child) {
        parent.appendChild(
            typeof child === 'string' ? document.createTextNode(child) : child);
      }
    }
  }
}

/** @const {!GoogDom} Helper object for DOM utils */
const googdom = new GoogDOM;

/**
 * An object that encapsulates one active evaluation.
 *
 * After constructing, setUpEval() should be called. After finishing one
 *   eval, clear() should be called. A new AntheaEval object should be
 *   used for every new eval.
 */
class AntheaEval {
  /**
   * @param {?AntheaManager} manager Optional AntheaManager controlling object.
   * @param {boolean=} readOnly Set to true when only reviewing an existing eval.
   */
  constructor(manager, readOnly=false) {
    /** @private ?AntheaManager */
    this.manager_ = manager;
    /** const boolean */
    this.READ_ONLY = readOnly;

    /** !Array<Element> */
    this.srcSpans = [];
    /** !Array<string> */
    this.srcSpansHTML = [];
    /** !Array<Element> */
    this.tgtSpans = [];
    /** !Array<string> */
    this.tgtSpansHTML = [];
    /** !Array<string> */
    this.tgtRealSpansHTML = [];

    /** number */
    this.currSentenceGroup = 0;

    /** ?Object */
    this.config = null;

    /** @private @const {number} */
    this.hotwPercent_ = 0;
    /** @private {!Array<!Object>} */
    this.evalResults_ = [];
    /** @private {!Object} */
    this.evalCounters_ = {};
    /** @private {number} */
    this.maxSentenceGroupShown_ = 0;

    /** @private @const {string} */
    this.noBlurColor_ = 'black';
    /** @private @const {string} */
    this.blurColor_ = 'lightgray';
    /** @private @const {string} */
    this.buttonColor_ = 'azure';

    /** @private {?Object} The currently marked phrase. */
    this.markedPhrase_ = null;
    /** @private {boolean} */
    this.startedMarking_ = false;
    /** @private {?PhraseMarker} */
    this.phraseMarker_ = null;
    /** @private {?Element} */
    this.evalPanel_ = null;
    /** @private {?Element} */
    this.prevButton_ = null;
    /** @private {?Element} */
    this.nextButton_ = null;

    /** @private {?Element} */
    this.contextRow_ = null;
    /** @private {?Element} */
    this.expanderRow_ = null;
    /** @private {?Element} */
    this.expanderText_ = null;
    /** @private {number} */
    this.numPrecedingVisible_ = 0;

    /** @private {!Array<!Object>} stored details for sentence groups */
    this.sentGroups_ = [];

    /**
     * @private {!Array<{eval: !Element, row: !Element,
     *                   startSG: number, numSG: number}>}
     */
    this.docs_ = [];
    /** @private {number} */
    this.currDoc_ = 0;
    /** @private {?Element} */
    this.prevDocButton_ = null;
    /** @private {?Element} */
    this.nextDocButton_ = null;
    /** @private {?Element} */
    this.displayedDocNum_ = null;
    /** @private {number} */
    this.numWordsEvaluated_ = 0;
    /** @private {number} */
    this.numTgtWords_ = 0;
    /** @private {?Element} */
    this.displayedProgress_ = null;

    /** {!Array<!Object>} */
    this.keydownListeners = [];

    /** number */
    this.lastTimestampMS_ = Date.now();

    /** function */
    this.resizeListener_ = null;
  }

  /**
   * Removes all window/document-level listeners and sets manager_ to null;
   */
  clear() {
    if (this.resizeListener_) {
      window.removeEventListener('resize', this.resizeListener_);
      this.resizeListener_ = null;
    }
    for (let listener of this.keydownListeners) {
      document.removeEventListener('keydown', listener);
    }
    this.keydownListeners = [];
    this.manager_ = null;
  }

  /**
   * Saves eval results to the manager_.
   */
  saveResults() {
    if (this.manager_ && !this.READ_ONLY) {
      this.manager_.persistActiveResults(this.evalResults_);
    }
  }

  /**
   * Restores eval results from the previously persisted value.
   */
  restoreResults(projectResults) {
    if (!this.manager_) {
      return;
    }
    if (!projectResults || projectResults.length == 0) {
      if (this.READ_ONLY) {
        this.manager_.log(this.manager_.ERROR,
                          'Cannot have a read-only eval when there are no ' +
                          'previous results to use');
      } else {
        this.manager_.log(this.manager_.INFO, 'No previous results to restore');
      }
      return;
    }
    if (projectResults.length != this.evalResults_.length) {
      this.manager_.log(
          this.manager_.ERROR,
          'Not restoring previous results as they are for ' +
          projectResults.length +
          ' sentence groups, but the current project has ' +
          this.evalResults_.length);
      return;
    }
    this.docs_[this.currDoc_].row.style.display = 'none';
    this.expanderRow_.style.display = 'none';

    this.evalResults_ = projectResults;
    this.currSentenceGroup = 0;
    this.numWordsEvaluated_ = this.numTgtWords_[0];
    while (this.currSentenceGroup + 1 < this.evalResults_.length &&
           (this.READ_ONLY ||
            this.evalResults_[this.currSentenceGroup + 1].visited)) {
      this.currSentenceGroup++;
      this.numWordsEvaluated_ += this.numTgtWords_[this.currSentenceGroup];
    }
    this.maxSentenceGroupShown_ = this.currSentenceGroup;

    if (this.READ_ONLY) {
      this.currSentenceGroup = 0;
    }
    this.currDoc_ = this.evalResults_[this.currSentenceGroup].doc;

    this.docs_[this.currDoc_].row.style.display = '';
    this.displayedDocNum_.innerHTML = '' + (this.currDoc_ + 1);
    const docEvalCell = this.docs_[this.currDoc_].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.displayedProgress_.innerHTML = this.getPercentEvaluated();
    this.showPageContextIfPresent();
    this.redrawAllSentenceGroups();
    this.recomputeTops();
    this.showCurrSentenceGroup();

    this.manager_.log(this.manager_.INFO,
                      'Restored previous evaluation results');
  }

  /**
   * Returns the evalResults_[] entry for the current sentence group.
   * @return {!Object}
   */
  currSentenceEval() {
    return this.evalResults_[this.currSentenceGroup];
  }

  /**
   * Records time used for the given action type, in the current evalResults_.
   */
  noteTiming(action) {
    if (this.READ_ONLY) {
      return;
    }
    const currEval = this.currSentenceEval();
    if (!currEval) {
      return;
    }
    const timing = currEval.timing;
    if (!timing) {
      return;
    }
    if (!timing[action]) {
      timing[action] = {
        count: 0,
        timeMS: 0,
      };
    }
    const tinfo = timing[action];
    tinfo.count++;
    currEval.timestamp = Date.now();
    tinfo.timeMS += (currEval.timestamp - this.lastTimestampMS_);
    this.lastTimestampMS_ = currEval.timestamp;
    this.saveResults();
  }

  /**
   * Shows the sentence group at index currSentenceGroup and enables buttons.
   */
  showCurrSentenceGroup() {
    this.redrawSentenceGroup(this.currSentenceGroup);
    this.setEvalButtonsAvailability();
  }

  /**
   * Returns true iff n is the current document's sentence group range.
   * @param {number} n The sentence group index to be tested.
   * @return {boolean}
   */
  inCurrDoc(n) {
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    return n >= start && n < start + num;
  }

  /**
   * Shows the sentence group at index n. How the sentence group gets shown
   *     depends on whether it is before, at, or after this.currSentenceGroup.
   * @param {number} n
   */
  redrawSentenceGroup(n) {
    if (!this.inCurrDoc(n)) {
      return;
    }

    /* Get rid of any old highlighting or listeners */
    const sg = this.sentGroups_[n];
    if (sg.clickListener) {
      this.srcSpans[n].removeEventListener('click', sg.clickListener);
      this.tgtSpans[n].removeEventListener('click', sg.clickListener);
    }
    sg.clickListener = null;

    const evalResult = this.evalResults_[n];

    googdom.setInnerHtml(this.srcSpans[n], this.srcSpansHTML[n]);
    this.srcSpans[n].className = 'anthea-source-sent-group';

    let tgtHTML = this.tgtSpansHTML[n];
    if (!this.READ_ONLY &&
        evalResult.hotw && !evalResult.hotw.done) {
      tgtHTML = this.tgtRealSpansHTML[n];
    }
    googdom.setInnerHtml(this.tgtSpans[n], tgtHTML);
    this.tgtSpans[n].className = 'anthea-target-sent-group';

    const srcWordSpans = this.srcSpans[n].getElementsByTagName('SPAN');
    const tgtWordSpans = this.tgtSpans[n].getElementsByTagName('SPAN');

    for (let error of evalResult.errors) {
      const severity = this.config.severities[error.severity];
      const color = severity.color;
      const spanArray = (error.location == 'source') ?
          srcWordSpans : tgtWordSpans;
      for (let x = error.start; x <= error.end; x++) {
        spanArray[x].style.backgroundColor = color;
      }
    }
    if (this.config.sqm) {
      const ratingInfo = this.config.sqm[evalResult.sqm];
      if (ratingInfo) {
        this.tgtSpans[n].style.textDecoration = 'underline';
        this.tgtSpans[n].style.textDecorationColor = ratingInfo.color;
      }
    }

    if (n == this.currSentenceGroup) {
      this.evalPanel_.style.top = sg.top;
      if (!this.config.sqm) {
        this.evalPanelErrors_.innerHTML = '';
        if (evalResult.hotw && evalResult.hotw.done) {
          this.displayHOTWMessage(evalResult.hotw.found,
                                  evalResult.hotw.injected_error);
        }
        for (let i = 0; i < evalResult.errors.length; i++) {
          this.displayError(evalResult.errors, i);
        }
        if (this.markedPhrase_) {
          const color =
              this.severity_ ? this.severity_.color : this.markedPhrase_.color;
          const spanArray = (this.markedPhrase_.location == 'source') ?
              srcWordSpans :
              tgtWordSpans;
          for (let x = this.markedPhrase_.start; x <= this.markedPhrase_.end;
               x++) {
            spanArray[x].style.backgroundColor = color;
          }
        }
      } else {
        const selected = evalResult.hasOwnProperty('sqm') ?
            evalResult.sqm : Infinity;
        for (let ratingInfo of this.config.sqm) {
          ratingInfo.button.style.color = (selected == ratingInfo.value) ?
            'blue' : 'black';
        }
      }
      this.noteTiming('visited-or-redrawn');
    }

    if ((n <= this.maxSentenceGroupShown_) && (n != this.currSentenceGroup) &&
        !this.startedMarking_) {
      // anthea-sent-group-nav class makes the mouse a pointer on hover.
      this.srcSpans[n].classList.add('anthea-sent-group-nav');
      this.tgtSpans[n].classList.add('anthea-sent-group-nav');
      sg.clickListener = () => {
        this.revisitSentenceGroup(n);
      };
      this.srcSpans[n].addEventListener('click', sg.clickListener);
      this.tgtSpans[n].addEventListener('click', sg.clickListener);
    }

    const blurSrc = n > this.currSentenceGroup;
    const blurTgt = n > this.currSentenceGroup;
    const boldSrcTgt = n == this.currSentenceGroup;
    this.srcSpans[n].style.color =
      blurSrc ? this.blurColor_ : this.noBlurColor_;
    this.tgtSpans[n].style.color =
      blurTgt ? this.blurColor_ : this.noBlurColor_;
    this.srcSpans[n].style.fontWeight = boldSrcTgt ? 'bold' : 'normal';
    this.tgtSpans[n].style.fontWeight = boldSrcTgt ? 'bold' : 'normal';

    if (n <= this.maxSentenceGroupShown_) {
      this.updateProgressForSentenceGroup(n);
    }
  }

  /**
   * Redraws all sentence groups and calls setEvalButtonsAvailability().
   */
  redrawAllSentenceGroups() {
    for (let n = 0; n < this.srcSpans.length; n++) {
      this.redrawSentenceGroup(n);
    }
    this.setEvalButtonsAvailability();
    this.lastTimestampMS_ = Date.now();
  }

  /**
   * Displays a "hands-on-the-wheel message", telling the rater whether or not
   *     they successfully found a deliberately injected HOTW error.
   * @param {boolean} found Whether the error was detected
   * @param {string} span The injected error phrase
   */
  displayHOTWMessage(found, span) {
    const tr = document.createElement('tr');
    this.evalPanelErrors_.appendChild(tr);
    let html = '<td class="anthea-eval-panel-text" colspan="2">';
    if (!found) {
      html += '<p class="anthea-hotw-missed">You missed some injected ' +
              'error(s) in this ';
    } else {
      html += '<p class="anthea-hotw-found">You successfully found an error ' +
              'in this ';
    }
    html +=
        '<span class="anthea-hotw-def" title="A small fraction of ' +
        'sentences that are initially shown with some deliberately injected ' +
        'error(s). Evaluating translation quality is a difficult and ' +
        'demanding task, and these test sentences are simply meant to ' +
        'help you maintain the high level of attention the task needs.">' +
        'test sentence group that had been artificially altered</span>.</p> ';
    html += '<p>The injected error (now <b>reverted</b>) was: ' + span + '</p>';
    html += '<p><b>Please continue to rate the translation as now shown ' +
            'without alterations, thanks!</b></p></td>';
    googdom.setInnerHtml(tr, html);
}

/**
 * Displays the previously marked error in errors[index], alongside the
 *     current sentence group. The displayed error also includes an "x" button
 *     for deleting it.
 * @param {!Array<!Object>} errors
 * @param {number} index
 */
displayError(errors, index) {
  const error = errors[index];
  const tr = document.createElement('tr');
  this.evalPanelErrors_.appendChild(tr);

  const delButton = googdom.createDom(
      'button', {class: 'anthea-stretchy-button anthea-eval-panel-short'}, '×');
  tr.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-nav',
      googdom.createDom('div', 'anthea-eval-panel-nav', delButton)));

  const severity = this.config.severities[error.severity];
  let desc = error.display || severity.display;
  if (error.subtype) {
    const errorInfo = this.config.errors[error.type] || {};
    if (errorInfo.subtypes && errorInfo.subtypes[error.subtype]) {
      desc = desc + ': ' + errorInfo.subtypes[error.subtype].display;
    }
  }
  if (error.metadata && error.metadata.note) {
    desc = desc + ' [' + error.metadata.note + ']';
  }
  desc += ': ';

  /**
   * Use 0-width spaces to ensure leading/trailing spaces get shown.
   */
  tr.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-text', desc,
      googdom.createDom(
          'span', {
            style: 'background-color:' + severity.color,
          },
          '\u200b' + error.selected + '\u200b')));

  delButton.title = 'Delete this error';
  if (this.READ_ONLY) {
    delButton.disabled = true;
  } else {
    delButton.addEventListener('click', (e) => {
      e.preventDefault();
      errors.splice(index, 1);
      this.showCurrSentenceGroup();
    });
  }
  }

  /**
   * Called from the PhraseMarker object, this is set when a phrase-start
   *     has been marked.
   */
  setStartedMarking() {
    this.startedMarking_ = true;
    this.setEvalButtonsAvailability();
  }

  /**
   * Displays the passed guidance message.
   * @param {string} text The guidance message.
   */
  showGuidance(text) {
    if (!this.guidance_) {
      return;
    }
    this.guidance_.innerHTML = text;
    this.guidance_.style.display = '';
  }

  /**
   * Sets the disabled/display state of all evaluation buttons appropriately.
   *    This is a critical function, as it determines, based upon the current
   *    state, which UI controls/buttons get shown and enabled.
   */
  setEvalButtonsAvailability() {
    const evalResult = this.currSentenceEval();
    const disableMarking = this.READ_ONLY ||
        (evalResult.errors && evalResult.errors.length > 0 &&
         (evalResult.errors[0].override_all_errors ||
          (this.config.MAX_ERRORS > 0 &&
           evalResult.errors.length >= this.config.MAX_ERRORS)));
    const disableSevErr =
        disableMarking || (this.config.MARK_SPAN_FIRST && !this.markedPhrase_);
    const disableSeverity = disableSevErr || this.severity_;
    for (let s in this.config.severities) {
      const severity = this.config.severities[s];
      severity.button.disabled = disableSeverity;
    }

    const disableErrors = disableSevErr || this.errorType_ ||
        (!this.config.MARK_SPAN_FIRST && !this.markedPhrase_);
    const location = this.markedPhrase_ ? this.markedPhrase_.location : '';
    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type]
      errorInfo.button.disabled = disableErrors;
      if (!disableErrors) {
        if (errorInfo.source_side_only && location && location != 'source') {
          errorInfo.button.disabled = true;
        }
        if (!errorInfo.source_side_ok && location && location == 'source') {
          errorInfo.button.disabled = true;
        }
        if (errorInfo.override_all_errors &&
            this.severityId_ && this.severityId_ != 'major') {
          errorInfo.button.disabled = true;
        }
        if (errorInfo.forced_severity && this.severityId_ &&
            this.severityId_ != errorInfo.forced_severity) {
          errorInfo.button.disabled = true;
        }
      }
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        subtypeInfo.button.disabled = disableErrors;
        if (!disableErrors) {
          if (subtypeInfo.source_side_only &&
              location && location != 'source') {
            subtypeInfo.button.disabled = true;
          }
          if (!subtypeInfo.source_side_ok && location && location == 'source') {
            subtypeInfo.button.disabled = true;
          }
        }
      }
    }
    this.expanderRow_.className = 'anthea-expander-row';
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    this.prevButton_.disabled = (this.currSentenceGroup <= start);
    this.nextButton_.disabled = (this.currSentenceGroup >= start + num - 1);
    if (this.config.sqm && !evalResult.hasOwnProperty('sqm') &&
        !this.READ_ONLY) {
      this.nextButton_.disabled = true;
    }
    this.prevDocButton_.style.display = (this.currDoc_ == 0) ? 'none' : '';
    this.prevDocButton_.disabled = false;
    if (this.currDoc_ == this.docs_.length - 1) {
      this.nextDocButton_.style.display = 'none';
    } else {
      this.nextDocButton_.style.display = '';
      this.nextDocButton_.disabled = !this.READ_ONLY &&
          !this.currDocFullyEvaluated();
    }
    if (this.config.sqm) {
      return;
    }

    // MQM-specific:
    this.guidancePanel_.style.backgroundColor =
        (this.severity_ ? this.severity_.color : 'whitesmoke');
    this.guidance_.style.display = 'none';
    this.evalPanelBody_.style.display = 'none';
    this.cancel_.style.display = 'none';
    this.evalPanelErrors_.style.display = '';
    if (this.READ_ONLY) {
      return;
    }
    if (!this.startedMarking_) {
      if (this.config.MARK_SPAN_FIRST) {
        if (!disableMarking) {
          this.phraseMarker_.getMarkedPhrase();
        }
      }
      return;
    }

    // MQM, and marking has already been initiated.
    this.evalPanelBody_.style.display = '';
    this.cancel_.style.display = '';
    this.evalPanelErrors_.style.display = 'none';

    if (this.config.MARK_SPAN_FIRST) {
      if (this.severity_) {
        this.showGuidance('Choose error type / sybtype');
      } else if (this.errorType_) {
        this.showGuidance('Choose error severity');
      } else {
        this.showGuidance('Choose error severity, type / subtype');
      }
    } else {
      if (!this.markedPhrase_ && this.severity_) {
        this.phraseMarker_.getMarkedPhrase(this.severity_.color);
      }
      if (this.markedPhrase_ && this.severity_) {
        this.showGuidance('Choose error type / sybtype');
      }
    }
    this.openSubtypes(null);
    this.prevButton_.disabled = true;
    this.nextButton_.disabled = true;
    this.prevDocButton_.disabled = true;
    this.nextDocButton_.disabled = true;
    this.expanderRow_.className = 'anthea-expander-row-disabled';
  }

  /**
   * Decrements currSentenceGroup within the current doc.
   */
  decrCurrSentenceGroup() {
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    if (this.currSentenceGroup <= start ||
        this.currSentenceGroup > start + num - 1) {
      return;
    }
    this.currSentenceGroup--;
  }

  /**
   * Updates displayed progress when sentence group n is visited for the first
   * time.
   * @param {number} n
   */
  updateProgressForSentenceGroup(n) {
    if (this.evalResults_[n].visited) {
      return;
    }
    this.evalResults_[n].visited = true;
    this.saveResults();
    this.numWordsEvaluated_ += this.numTgtWords_[n];
    this.displayedProgress_.innerHTML = this.getPercentEvaluated();
  }

  /**
   * Called after a sentence group should be done with. Returns false in
   *     the (rare) case that the sentence group was a HOTW sentence with
   *     injected errors shown, which leads to the end of the HOTW check
   *     but makes the rater continue to rate the sentence.
   * @return {boolean}
   */
  finishCurrSentenceGroup() {
    const evalResult = this.evalResults_[this.currSentenceGroup];
    if (!this.READ_ONLY &&
        evalResult.hotw && !evalResult.hotw.done) {
      this.noteTiming('missed-hands-on-the-wheel-error');
      evalResult.hotw.done = true;
      evalResult.hotw.timestamp = evalResult.timestamp;
      evalResult.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.redrawAllSentenceGroups();
      return false;
    }
    return true;
  }

  /**
   * Increments currSentenceGroup within the current doc, keeping track of its
   *     max value so far.
   */
  incrCurrSentenceGroup() {
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    if (this.currSentenceGroup < start ||
        this.currSentenceGroup >= start + num - 1) {
      return;
    }
    if (!this.finishCurrSentenceGroup()) {
      return;
    }
    this.currSentenceGroup++;
    if (this.currSentenceGroup > this.maxSentenceGroupShown_) {
      this.maxSentenceGroupShown_ = this.currSentenceGroup;
    }
  }

  /**
   * Returns true if the current sentence group is the farthest one seen.
   * @return {boolean}
   */
  isFarthestSentenceGroup() {
    return this.currSentenceGroup == this.maxSentenceGroupShown_;
  }

  /**
   * Navigates to the previous sentence group.
   */
  handlePrev() {
    this.decrCurrSentenceGroup();
    this.redrawAllSentenceGroups();
  }

  /**
   * Navigates to the next sentence group.
   */
  handleNext() {
    this.incrCurrSentenceGroup();
    this.redrawAllSentenceGroups();
  }

  /**
   * Gets the percentage of words evaluated.
   * @return {string}
   */
  getPercentEvaluated() {
    return '' + Math.min(
      100,
      Math.floor(100 * this.numWordsEvaluated_ / this.evalCounters_.numWords)) +
      '%';
  }

  /**
   * If the passed error requires additional user input, this function
   *     augments it with the needed info. Returns false if the user cancels.
   * @param {!Object} markedError Object with error details.
   * @return {boolean} Whether to continue with marking the error.
   */
  maybeAugmentError(markedError) {
    if (markedError.override_all_errors) {
      if (!confirm('This error category will remove all other marked errors ' +
                   'from this sentence group and will set the error span to ' +
                   'be the whole sentence group. Please confirm!')) {
        this.noteTiming('cancelled-override-all-errors');
        return false;
      }
      this.noteTiming('confirmed-override-all-errors');
      markedError.location = 'translation';
      markedError.prefix = '';
      const spanArray =
          this.tgtSpans[this.currSentenceGroup].getElementsByTagName('SPAN');
      markedError.selected = '';
      for (let x = 0; x < spanArray.length; x++) {
        markedError.selected += spanArray[x].innerText;
      }
      markedError.start = 0;
      markedError.end = spanArray.length - 1;
    }

    if (!markedError.metadata) {
      markedError.metadata = {};
    }
    if (markedError.needs_note) {
      markedError.metadata.note = prompt(
          "Please enter a short error description", "");
      if (!markedError.metadata.note) {
        this.noteTiming('cancelled-error-note');
        return false;
      }
      this.noteTiming('added-error-note');
    }
    return true;
  }

  /**
   * Calling this marks the end of an error-marking in the current sentence
   *     group.
   * @param {?Object} markedError Object with error details and location,
   *     or null, if no error is to be marked.
   */
  concludeMarkingPhrase(markedError) {
    if (markedError) {
      const evalResult = this.currSentenceEval();
      if (markedError.override_all_errors) {
        evalResult.errors = [];
      }
      if (!markedError.metadata) {
        markedError.metadata = {};
      }
      markedError.metadata.timestamp = evalResult.timestamp;
      markedError.metadata.timing = evalResult.timing;
      evalResult.timing = {};
      evalResult.errors.push(markedError);
      this.displayError(evalResult.errors, evalResult.errors.length - 1);
    }
    this.resetMQMRating();

    this.saveResults();
    this.redrawAllSentenceGroups();
  }

  /**
   * Marks a "full-span" error that overrides all other errors and bypasses
   *     phrase selection.
   */
  markFullSpanError(severityId) {
    const severity = this.config.severities[severityId];
    const error = {
      severity: severityId,
      override_all_errors: true,
      needs_note: severity.full_span_error &&
                  severity.full_span_error.needs_note ? true : false,
      metadata: {},
    };
    if (!this.maybeAugmentError(error)) {
      this.concludeMarkingPhrase(null);
      return;
    }

    const evalResult = this.currSentenceEval();
    if (evalResult.hotw && !evalResult.hotw.done) {
      this.noteTiming('found-hands-on-the-wheel-error');
      evalResult.hotw.done = true;
      evalResult.hotw.found = true;
      evalResult.hotw.timestamp = evalResult.timestamp;
      evalResult.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.concludeMarkingPhrase(null);
      return;
    }

    this.concludeMarkingPhrase(error);
  }

  /**
   * Opens the visibility of the list of subtypes for the current error type,
   *     closing all others.
   * @param {?Element} subtypes If not-null, open this subtypes panel.
   */
  openSubtypes(subtypes) {
    if (this.openSubtypes_) {
      this.openSubtypes_.style.display = 'none';
      if (subtypes == this.openSubtypes_) {
        this.openSubtypes_ = null;
        return;
      }
    }
    if (subtypes) {
      subtypes.style.display = '';
      this.openSubtypes_ = subtypes;
    }
  }

  /**
   * Creates the eval panel shown in the evaluation column.
   */
  makeEvalPanel() {
    if (this.config.sqm) {
      this.makeEvalPanelSQM();
    } else {
      this.makeEvalPanelMQM();
    }
  }

  /**
   * Creates the SQM eval panel shown in the evaluation column.
   */
  makeEvalPanelSQM() {
    const ratingsButtons = googdom.createDom('table',
                                             'anthea-eval-panel-table');
    this.evalPanelBody_.appendChild(ratingsButtons);

    for (let ratingInfo of this.config.sqm) {
      const ratingValue =
          googdom.createDom('div',
                            'anthea-rating-value', '' + ratingInfo.value);
      const ratingDisplay =
          googdom.createDom('div',
                            'anthea-rating-display', ratingInfo.display || '');
      ratingDisplay.style.textDecorationColor = ratingInfo.color;
      const ratingButton = googdom.createDom(
          'tr', 'anthea-rating-button',
          googdom.createDom('td', null, ratingValue),
          googdom.createDom('td', null, ratingDisplay));
      if (ratingInfo.description) {
        ratingButton.title = ratingInfo.description;
      }
      ratingsButtons.appendChild(ratingButton);
      if (!this.READ_ONLY) {
        const listener = (e) => {
          if (e.type == 'click' || (e.key && e.key == ratingInfo.shortcut)) {
            e.preventDefault();
            this.setSQMRating(ratingInfo.value);
          }
        };
        ratingButton.addEventListener('click', listener);
        this.keydownListeners.push(listener);
        document.addEventListener('keydown', listener);
      } else {
        ratingButton.classList.add('anthea-rating-button-disabled');
      }
      ratingInfo.button = ratingButton;
    }
  }

  /**
   * Creates the MQM eval panel shown in the evaluation column.
   */
  makeEvalPanelMQM() {
    this.guidancePanel_ = googdom.createDom('div', 'anthea-guidance-panel');
    this.evalPanelHead_.appendChild(this.guidancePanel_);

    this.guidance_ = googdom.createDom('div', 'anthea-eval-guidance');
    this.guidance_.style.display = 'none';
    this.guidancePanel_.appendChild(this.guidance_);

    this.cancel_ = googdom.createDom(
        'button', 'anthea-stretchy-button anthea-eval-cancel', 'Cancel (Esc)');
    this.cancel_.style.display = 'none';
    const cancelListener = (e) => {
      if (e.type == 'click' ||
          (!this.cancel_.disabled && e.key && e.key === 'Escape')) {
        e.preventDefault();
        this.handleCancel();
      }
    };
    this.cancel_.addEventListener('click', cancelListener);
    this.keydownListeners.push(cancelListener);
    document.addEventListener('keydown', cancelListener);
    this.guidancePanel_.appendChild(this.cancel_);

    this.evalPanelErrorTypes_ = googdom.createDom(
        'table', 'anthea-eval-panel-table');
    this.evalPanelBody_.appendChild(this.evalPanelErrorTypes_);
    this.openSubtypes_ = null;

    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type];
      const errorButton = googdom.createDom(
          'button', 'anthea-error-button',
          errorInfo.display + (errorInfo.subtypes ? ' ▶' : ''));
      if (errorInfo.description) {
        errorButton.title = errorInfo.description;
      }
      const errorCell = googdom.createDom(
          'td', 'anthea-eval-panel-cell', errorButton);
      if (!errorInfo.hidden) {
        // We add the button to the DOM only if not hidden.
        this.evalPanelErrorTypes_.appendChild(
            googdom.createDom('tr', null, errorCell));
      }
      if (errorInfo.subtypes) {
        errorButton.style.background = 'lightblue';
        const subtypesDiv = googdom.createDom('div',
                                              'anthea-eval-panel-subtypes');
        errorCell.appendChild(subtypesDiv);
        const subtypes = googdom.createDom('table',
                                           'anthea-eval-panel-table');
        subtypesDiv.appendChild(subtypes);
        for (let subtype in errorInfo.subtypes) {
          let subtypeInfo = errorInfo.subtypes[subtype];
          const display = this.config.FLATTEN_SUBTYPES ?
              errorInfo.display + ' / ' + subtypeInfo.display :
              subtypeInfo.display;
          const subtypeButton =
              googdom.createDom('button', 'anthea-error-button', display);
          if (subtypeInfo.description) {
            subtypeButton.title = subtypeInfo.description;
          }
          subtypeButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.setMQMType(type, subtype);
          });
          subtypeInfo.button = subtypeButton;
          if (!subtypeInfo.hidden) {
            // We add the button to the DOM only if not hidden.
            subtypes.appendChild(googdom.createDom(
                'tr', null,
                googdom.createDom(
                    'td', 'anthea-eval-panel-cell', subtypeButton)));
          }
        }
        if (this.config.FLATTEN_SUBTYPES) {
          errorButton.style.display = 'none';
        } else {
          subtypesDiv.className =
              subtypesDiv.className + ' anthea-eval-panel-unflattened';
          subtypes.style.display = 'none';
          errorButton.addEventListener('click', (e) => {
            e.preventDefault();
            this.openSubtypes(subtypes);
          });
        }
      } else {
        errorButton.addEventListener('click', (e) => {
          e.preventDefault();
          this.setMQMType(type);
        });
      }
      errorButton.disabled = true;
      errorInfo.button = errorButton;
    }
  }

  /**
   * Resets the current MQM rating.
   */
  resetMQMRating() {
    this.severityId_ = '';
    this.severity_ = null;
    this.errorType_ = '';
    this.errorSubtype_ = '';
    this.markedPhrase_ = null;
    this.startedMarking_ = false;
    this.evalPanelBody_.style.display = 'none';
  }

  /**
   * Called after any of the three parts of an MQM rating (error span,
   *     severity, error type) is added, this finishes the rating when
   *     all three parts have been received.
   */
  maybeSetMQMRating() {
    this.startedMarking_ = true;
    this.redrawAllSentenceGroups();
    if (!this.severity_ ||
        (!this.errorType_ && !this.severity_.forced_error) ||
        !this.markedPhrase_) {
      return;
    }
    const errorInfo = this.config.errors[this.errorType_] ||
        this.severity_.forced_error || {};
    const display =
        errorInfo.display || this.severity_.display || 'Unspecified';
    const error = {
      location: this.markedPhrase_.location,
      prefix: this.markedPhrase_.prefix,
      selected: this.markedPhrase_.selected,
      type: this.errorType_,
      subtype: this.errorSubtype_,
      display: display,
      start: this.markedPhrase_.start,
      end: this.markedPhrase_.end,
      severity: this.severityId_,
      override_all_errors: errorInfo.override_all_errors ? true : false,
      needs_note: errorInfo.needs_note ? true : false,
      metadata: {},
    };
    if (!this.maybeAugmentError(error)) {
      this.concludeMarkingPhrase(null);
      return;
    }
    this.concludeMarkingPhrase(error);
  }

  /**
   * Sets the severity level for the current MQM rating.
   * @param {string} severityId The severity of the error.
   */
  setMQMSeverity(severityId) {
    this.severityId_ = severityId;
    this.severity_ = this.config.severities[severityId];
    this.noteTiming('chose-severity-' + severityId);
    if (this.markedPhrase_ && this.severity_.forced_error) {
      this.setMQMType('');
      return;
    }
    this.maybeSetMQMRating();
  }

  /**
   * Sets the MQM error type and subtype for the current MQM rating.
   * @param {string} type
   * @param {string=} subtype
   */
  setMQMType(type, subtype = '') {
    this.errorType_ = type;
    this.errorSubtype_ = subtype;
    this.noteTiming('chose-error-' + type + (subtype ? '-' + subtype : ''));
    const errorInfo = this.config.errors[type];
    if (errorInfo.override_all_errors) {
      this.setMQMSeverity('major');
      return;
    }
    if (errorInfo.forced_severity) {
      this.setMQMSeverity(errorInfo.forced_severity);
      return;
    }
    this.maybeSetMQMRating();
  }

  /**
   * Sets the MQM error span for the current MQM rating.
   * @param {!Object} markedPhrase
   */
  setMQMPhrase(markedPhrase) {
    this.markedPhrase_ = markedPhrase;
    const evalResult = this.currSentenceEval();
    if (evalResult.hotw && !evalResult.hotw.done) {
      this.noteTiming('found-hands-on-the-wheel-error');
      evalResult.hotw.done = true;
      evalResult.hotw.found = true;
      evalResult.hotw.timestamp = evalResult.timestamp;
      evalResult.hotw.timing = evalResult.timing;
      evalResult.timing = {};
      this.concludeMarkingPhrase(null);
      return;
    }
    this.noteTiming('ended-marking-phrase');
    if (this.severity_ && this.severity_.forced_error) {
      this.setMQMType('');
      return;
    }
    this.showGuidance('Choose error type / subtype');
    this.maybeSetMQMRating();
  }

  /**
   * Set the SQM rating for this sentence.
   * @param {string} sqm rating (typically numeric, but need not be).
   */
  setSQMRating(sqm) {
    const evalResult = this.currSentenceEval();
    evalResult.sqm = sqm;
    this.saveResults();
    if (this.isFarthestSentenceGroup()) {
      this.incrCurrSentenceGroup();
    }
    this.redrawAllSentenceGroups();
  }

  /**
   * Handles cancellation of the current error-marking.
   */
  handleCancel() {
    if (this.config.sqm) {
      return;
    }
    this.noteTiming('cancelled-marking-error');
    this.concludeMarkingPhrase(null);
  }

  /**
   * Handles a click on a "<severity>" button.
   * @param {string} severityId
   */
  handleSeverityClick(severityId) {
    const severity = this.config.severities[severityId];
    if (severity.button.disabled) {
      return;
    }
    this.noteTiming('chose-severity-' + severityId);
    if (severity.full_span_error) {
      // Just mark the full-span error directly.
      this.markFullSpanError(severityId);
    } else {
      this.setMQMSeverity(severityId);
    }
  }

  /**
   * Navigates to the specified sentence group and opens highlighting UI.
   * @param {number} n
   */
  revisitSentenceGroup(n) {
    if (n > this.maxSentenceGroupShown_ || n < 0 || !this.inCurrDoc(n) ||
        this.startedMarking_) {
      return;
    }
    this.noteTiming('revisited');
    this.evalCounters_.numRevisits++;
    let old = this.currSentenceGroup;
    this.currSentenceGroup = n;
    let incr = 1;
    if (old > n) {
      incr = -1;
    }
    while (old != n) {
      this.redrawSentenceGroup(old);
      old = old + incr;
    }
    this.showCurrSentenceGroup();
  }

  /**
   * Returns a random integer in the range [0, max).
   * @param {number} max
   * @return {number}
   */
  static getRandomInt(max) {
    return Math.floor(Math.random() * max);
  }

  /**
   * If the text is sufficiently long, then this function injects a deliberate
   *     translation error in some part of the text (currently it reverses a
   *     long-enough sub-phrase). The returned object includes a "corrupted"
   *     field that has the corrupted text (empty if no corruption was done),
   *     and a "display" field suitable for revealing after the corruption is
   *     undone.
   * @param {string} text
   * @return {!Object}
   */
  static injectErrors(text) {
    const ret = {
      corrupted: '',
      display: '',
    };
    /**
     * "pieces" will be an array of tokens, including text as well as separators
     * (spaces and 0-width spaces). Error injection is done by reversing a
     * segment from "pieces" that starts and ends on separators.
     */
    const pieces = [];
    let piece = '';
    const seps = [];
    for (let c of text) {
      if (c == ' ' || c == '\u200b') {
        if (piece) pieces.push(piece);
        seps.push(pieces.length);
        pieces.push(c);
        piece = '';
      } else {
        piece += c;
      }
    }
    if (piece) pieces.push(piece);

    if (seps.length <= 6) {
      // Too short.
      return ret;
    }
    // Start within the first half.
    const start = this.getRandomInt(seps.length / 2);
    const starti = seps[start];
    const end = Math.min(seps.length - 1, start + 4 + this.getRandomInt(4));
    const endi = seps[end];
    // Reverse
    ret.corrupted = pieces.slice(0, starti + 1).join('') +
      pieces.slice(starti + 1, endi).reverse().join('') +
      pieces.slice(endi).join('');
    ret.display = '<span class="anthea-hotw-revealed">' +
      pieces.slice(starti + 1, endi).reverse().join('') + '</span>';
    return ret;
  }

  /**
   * Wraps each space-separated word in text in a SPAN of class
   *   "anthea-word" and each space too in a SPAN of class
   *   "anthea-space".
   * @param {string} text
   * @return {{numWords: number, spannified: string}}
   */
  static spannifyWords(text) {
    const words = text.split(' ');
    const ret = {
      numWords: 0,
    };
    for (let i = 0; i < words.length; i++) {
      // Segment further by any 0-width space characters present.
      const cjkWords = words[i].split('\u200b');
      words[i] = '';
      for (let w of cjkWords) {
        words[i] += '<span class="anthea-word">' + w + '</span>';
      }
      ret.numWords += cjkWords.length;
    }
    const SEP = '<span class="anthea-space"> </span>';
    ret.spannified = words.join(SEP) + SEP;
    return ret;
  }

  /**
   * Show k sentences of preceding context.
   * @param {number} k
   */
  showPrecedingContext(k) {
    const srcPrec = document.getElementsByClassName("prec-src-group");
    const tgtPrec = document.getElementsByClassName("prec-tgt-group");
    this.numPrecedingVisible_ = 0;
    for (let i = 1; i <= srcPrec.length; i++) {
      const index = srcPrec.length - i;
      if (i <= k) {
        srcPrec[index].style.display = '';
        tgtPrec[index].style.display = '';
        this.numPrecedingVisible_++;
      } else {
        srcPrec[index].style.display = 'none';
        tgtPrec[index].style.display = 'none';
      }
    }
  }

  /**
   * If the first doc has more preceding context available, then this
   *     function toggles its display.
   */
  toggleExpansion() {
    if (this.startedMarking_ || this.expanderRow_.style.display == 'none') {
      return;
    }
    const extra =
        this.evalCounters_.numPreceding - this.config.NUM_PRECEDING_VISIBLE;
    if (extra <= 0) {
      return;
    }
    if (this.numPrecedingVisible_ > this.config.NUM_PRECEDING_VISIBLE) {
      this.showPrecedingContext(this.config.NUM_PRECEDING_VISIBLE);
      this.expanderText_.innerHTML =
          'Click here to expand ' + extra +
          ' more preceding context sentence group(s)';
    } else {
      this.showPrecedingContext(this.evalCounters_.numPreceding);
      this.expanderText_.innerHTML =
          'Click here to collapse ' + extra +
          ' preceding context sentence group(s)';
    }
    this.isExpanded_ = !this.isExpanded_;
    this.recomputeTops();
    this.noteTiming('toggled-context');
    this.evalCounters_.numFullContextToggles++;
  }

  /**
   * This function recomputes the tops of sentence groups.
   */
  recomputeTops() {
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    let docRowRect = this.docs_[this.currDoc_].row.getBoundingClientRect();
    let pos = 0;
    for (let s = start; s < start + num; s++) {
      const srcRect = this.srcSpans[s].getBoundingClientRect();
      const tgtRect = this.tgtSpans[s].getBoundingClientRect();
      pos = Math.min(srcRect.top - docRowRect.top,
                     tgtRect.top - docRowRect.top);
      const sg = this.sentGroups_[s];
      sg.top = '' + pos + 'px';
      if (s == this.currSentenceGroup && this.evalPanel_) {
        this.evalPanel_.style.top = sg.top;
      }
    }
    // Make sure the table height is sufficient.
    const docEvalCell = this.docs_[this.currDoc_].eval;
    docEvalCell.style.height = '' + (pos + 600) + 'px';
  }

  /**
   * Returns true if all sentence groups in the current doc have been evaluated.
   *
   * @return {boolean} Whether the current doc has been fully evaluated.
   */
  currDocFullyEvaluated() {
    const start = this.docs_[this.currDoc_].startSG;
    const num = this.docs_[this.currDoc_].numSG;
    for (let s = start; s < start + num; s++) {
      if (!this.evalResults_[s].visited) {
        return false;
      }
    }
    return true;
  }

  /**
   * Returns to the previous document.
   */
  prevDocument() {
    if (!this.READ_ONLY && (this.startedMarking_ || this.currDoc_ == 0)) {
      return;
    }
    if (!this.finishCurrSentenceGroup()) {
      return;
    }
    this.docs_[this.currDoc_].row.style.display = 'none';
    this.expanderRow_.style.display = 'none';
    this.currDoc_--;
    this.displayedDocNum_.innerHTML = '' + (this.currDoc_ + 1);
    this.currSentenceGroup = this.docs_[this.currDoc_].startSG;
    this.docs_[this.currDoc_].row.style.display = '';
    const docEvalCell = this.docs_[this.currDoc_].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showPageContextIfPresent();
    this.redrawAllSentenceGroups();
    this.recomputeTops();
    this.showCurrSentenceGroup();
  }

  /**
   * Proceeds to the next document.
   */
  nextDocument() {
    if (!this.READ_ONLY &&
        (this.startedMarking_ || this.currDoc_ == this.docs_.length - 1 ||
         !this.currDocFullyEvaluated())) {
      return;
    }
    if (!this.finishCurrSentenceGroup()) {
      return;
    }
    this.docs_[this.currDoc_].row.style.display = 'none';
    this.expanderRow_.style.display = 'none';
    this.currDoc_++;
    this.displayedDocNum_.innerHTML = '' + (this.currDoc_ + 1);
    this.currSentenceGroup = this.docs_[this.currDoc_].startSG;
    if (this.currSentenceGroup > this.maxSentenceGroupShown_) {
      this.maxSentenceGroupShown_ = this.currSentenceGroup;
    }
    this.docs_[this.currDoc_].row.style.display = '';
    const docEvalCell = this.docs_[this.currDoc_].eval;
    docEvalCell.appendChild(this.evalPanel_);
    this.showPageContextIfPresent();
    this.redrawAllSentenceGroups();
    this.recomputeTops();
    this.showCurrSentenceGroup();
  }

  /**
   * Populates an instructions panel with instructions and lists of severities
   * and error types for MQM.
   * @param {!Element} panel The DIV element to populate.
   */
  populateMQMInstructions(panel) {
    panel.innerHTML = (this.config.instructions || '') +
        (!this.config.SKIP_RATINGS_TABLES ? `
      <p>
        <details open>
          <summary>
            <b>List of severities:</b>
          </summary>
          <ul id="anthea-mqm-list-of-severities"></ul>
        </details>
      </p>
      <p>
        <details>
          <summary>
            <b>Table of error types:</b>
          </summary>
          <table id="anthea-mqm-errors-table" class="anthea-errors-table">
          </table>
        </details>
      </p>` : '');

    if (this.config.SKIP_RATINGS_TABLES) {
      return;
    }
    const listOfSeverities = document.getElementById(
        'anthea-mqm-list-of-severities');
    for (let s in this.config.severities) {
      const severity = this.config.severities[s];
      if (severity.hidden) {
        continue;
      }
      listOfSeverities.appendChild(googdom.createDom(
          'li', null,
          googdom.createDom(
            'span',
            { style: 'font-weight:bold; background-color:' + severity.color },
            severity.display),
          ': ' + severity.description));
    }
    const errorsTable = document.getElementById('anthea-mqm-errors-table');
    for (let type in this.config.errors) {
      const errorInfo = this.config.errors[type];
      if (errorInfo.hidden) {
        continue;
      }
      const errorLabel = googdom.createDom(
          'td', 'anthea-error-label', errorInfo.display);
      const errorDesc = googdom.createDom(
          'td', {colspan: 2}, errorInfo.description);
      errorsTable.appendChild(googdom.createDom(
          'tr', null, errorLabel, errorDesc));
      if (!errorInfo.subtypes) {
        continue;
      }
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        if (subtypeInfo.hidden) {
          continue;
        }
        const emptyCol = document.createElement('td');
        const subtypeLabel = googdom.createDom(
          'td', 'anthea-error-label', subtypeInfo.display);
        const subtypeDesc = googdom.createDom(
          'td', null, subtypeInfo.description);
        errorsTable.appendChild(googdom.createDom(
          'tr', null, emptyCol, subtypeLabel, subtypeDesc));
      }
    }
  }

  /**
   * Populates an instructions panel with list of ratings for SQM.
   * @param {!Element} panel The DIV element to populate.
   */
  populateSQMLabelsDescPanel(panel) {
    panel.innerHTML = (this.config.instructions || '') +
        (!this.config.SKIP_RATINGS_TABLES ? `
      <p>
        <details open>
          <summary>
            <b>Table of ratings:</b>
            <br><br>
          </summary>
          <table id="anthea-sqm-errors-table" class="anthea-errors-table">
          </table>
        </details>
      </p>` : '');

    if (this.config.SKIP_RATINGS_TABLES) {
      return;
    }
    const ratingsTable = document.getElementById('anthea-sqm-errors-table');
    for (let ratingInfo of this.config.sqm) {
      const ratingValue = googdom.createDom(
          'td', 'anthea-error-label', '' + ratingInfo.value);
      ratingValue.style.background = ratingInfo.color;
      const ratingDisplay = googdom.createDom(
          'td', 'anthea-error-label', ratingInfo.display);
      const ratingDesc = googdom.createDom(
          'td', null, ratingInfo.description);
      ratingsTable.appendChild(googdom.createDom(
          'tr', null, ratingValue, ratingDisplay, ratingDesc));
    }
  }

  /**
   * Modifies the config object, applying overrides.
   * @param {!Object} config The configuration object.
   * @param {string} overrides
   */
  applyConfigOverrides(config, overrides) {
    const parts = overrides.split(',');
    for (let override of parts) {
      override = override.trim();
      // Each override can look like:
      // +severity:<sev> or -severity:<sev>
      // +error:<type>[:<subtype>] or -error:<type>[:<subtype>]
      let shouldAdd = true;
      if (override.charAt(0) == '+') {
        shouldAdd = true;
      } else if (override.charAt(0) == '-') {
        shouldAdd = false;
      } else {
        continue;
      }
      const subparts = override.substr(1).split(':');
      if (subparts.length < 2) {
        continue;
      }
      if (subparts[0] == 'severity') {
        const severity = subparts[1];
        if (!config.severities[severity]) {
          continue;
        }
        config.severities[severity].hidden = !shouldAdd;
      } else if (subparts[0] == 'error') {
        const type = subparts[1];
        if (!config.errors[type]) {
          continue;
        }
        let errorInfo = config.errors[type];
        if (subparts.length > 2) {
          const subtype = subparts[2];
          if (!errorInfo.subtypes || !errorInfo.subtypes[subtype]) {
            continue;
          }
          errorInfo = errorInfo.subtypes[subtype];
        }
        errorInfo.hidden = !shouldAdd;
      } else {
        continue;
      }
    }
  }

  /**
   * Creates the UI. Copies config into this_.config.
   *
   * @param {!Object} config The configuration object.
   * @param {?Array<!Object>} projectResults Previously saved results or [].
   * @param {!Element} instructionsPanel The instructions panel to populate.
   * @param {!Element} controlPanel The control panel to populate.
   */
  createUIFromConfig(config, projectResults, instructionsPanel, controlPanel) {
    this.config = config;
    /* Remove previous keydown listeners if any */
    for (let listener of this.keydownListeners) {
      document.removeEventListener('keydown', listener);
    }
    this.keydownListeners = [];

    /* Show preceding sentences, if any */
    this.showPrecedingContext(config.NUM_PRECEDING_VISIBLE);
    if (this.numPrecedingVisible_ < this.evalCounters_.numPreceding) {
      this.expanderText_.innerHTML =
          'Click here to expand ' +
          (this.evalCounters_.numPreceding - this.numPrecedingVisible_)  +
          ' more preceding context sentence group(s)';
      this.expanderRow_.style = '';
    } else {
      this.expanderRow_.style.display = 'none';
    }

    const docEvalCell = this.docs_[this.currDoc_].eval;
    docEvalCell.innerHTML = '';
    this.evalPanel_ = googdom.createDom(
      'div', {id: 'anthea-eval-panel', class: 'anthea-eval-panel'});
    docEvalCell.appendChild(this.evalPanel_);

    if (!config.sqm) {
      this.populateMQMInstructions(instructionsPanel);
    } else {
      this.populateSQMLabelsDescPanel(instructionsPanel);
    }

    this.evalPanelHead_ = googdom.createDom('div', 'anthea-eval-panel-head');
    this.evalPanel_.appendChild(this.evalPanelHead_);

    this.evalPanelBody_ = googdom.createDom('div', 'anthea-eval-panel-body');
    this.evalPanelBody_.style.display = 'none';
    this.evalPanel_.appendChild(this.evalPanelBody_);

    this.evalPanelList_ = googdom.createDom('div', 'anthea-eval-panel-list');
    this.evalPanel_.appendChild(this.evalPanelList_);
    this.evalPanelErrors_ = googdom.createDom(
        'table', 'anthea-eval-panel-table');
    this.evalPanelList_.appendChild(this.evalPanelErrors_);

    const buttonsRow = document.createElement('tr');
    this.evalPanelHead_.appendChild(
      googdom.createDom('table', 'anthea-eval-panel-table', buttonsRow));

    this.prevButton_ = googdom.createDom(
      'button', { id: 'anthea-prev-button',
                  class: 'anthea-stretchy-button anthea-eval-panel-tall',
                  title: 'Go back to the previous sentence group ' +
                         '(shortcut: left-arrow key)' },
      '←');
    const prevListener = (e) => {
      if (e.type == 'click' ||
          (!this.prevButton_.disabled && e.key && e.key === "ArrowLeft")) {
        e.preventDefault();
        this.handlePrev();
      }
    };
    this.prevButton_.addEventListener('click', prevListener);
    this.keydownListeners.push(prevListener);
    document.addEventListener('keydown', prevListener);
    buttonsRow.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-nav',
      googdom.createDom('div', 'anthea-eval-panel-nav', this.prevButton_)));

    if (!config.sqm) {
      // MQM.
      for (let s in config.severities) {
        const severity = config.severities[s];
        const action = severity.action || severity.display;
        const buttonText =
            action + (severity.shortcut ? ' [' + severity.shortcut + ']' : '');
        severity.button = googdom.createDom(
            'button', {
              class: 'anthea-stretchy-button anthea-eval-panel-tall',
              style: 'background-color:' + severity.color,
              title: severity.description
            },
            buttonText);
        const listener = (e) => {
          if (e.type == 'click' || (e.key && e.key == severity.shortcut)) {
            e.preventDefault();
            this.handleSeverityClick(s);
          }
        };
        if (severity.shortcut) {
          this.keydownListeners.push(listener);
          document.addEventListener('keydown', listener);
        }
        severity.button.addEventListener('click', listener);
        if (!severity.hidden) {
          buttonsRow.appendChild(googdom.createDom(
              'td', 'anthea-eval-panel-cell', severity.button));
        }
      }
    } else {
      // SQM.
      buttonsRow.appendChild(googdom.createDom(
          'td', 'anthea-eval-panel-text anthea-bold',
          'Rating for translation of current sentence group:'));
    }

    this.nextButton_ = googdom.createDom(
      'button', { id: 'anthea-next-button',
                  class: 'anthea-stretchy-button anthea-eval-panel-tall',
                  title: 'Go to the next sentence group ' +
                         '(shortcut: right-arrow key)' },
      '→');
    const nextListener = (e) => {
      if (e.type == 'click' ||
          (!this.nextButton_.disabled && e.key && e.key === "ArrowRight")) {
        e.preventDefault();
        this.handleNext();
      }
    };
    this.nextButton_.addEventListener('click', nextListener);
    this.keydownListeners.push(nextListener);
    document.addEventListener('keydown', nextListener);
    buttonsRow.appendChild(googdom.createDom(
      'td', 'anthea-eval-panel-nav',
      googdom.createDom('div', 'anthea-eval-panel-nav', this.nextButton_)));

    this.displayedDocNum_ = googdom.createDom(
      'span', null, '' + (this.currDoc_ + 1));
    this.displayedProgress_ = googdom.createDom('span', 'anthea-bold',
                                                 this.getPercentEvaluated());
    const progressMessage = googdom.createDom(
        'span', 'anthea-status-text', 'Document no. ',
        this.displayedDocNum_, ' of ' + this.docs_.length);
    if (!this.READ_ONLY) {
      progressMessage.appendChild(googdom.createDom(
          'span', null, ' (across all documents, ', this.displayedProgress_,
          ' of translation text has been evaluated so far)'));
    }
    this.prevDocButton_ = googdom.createDom(
      'button', {
        id: 'anthea-prev-doc-button', class: 'anthea-docnav-eval-button',
        title: 'Revisit the previous document' },
      'Prev Document');
    this.prevDocButton_.style.backgroundColor = this.buttonColor_;
    this.prevDocButton_.addEventListener(
      'click', (e) => {
        e.preventDefault();
        this.prevDocument();
    });
    this.nextDocButton_ = googdom.createDom(
      'button', {
        id: 'anthea-next-doc-button', class: 'anthea-docnav-eval-button',
        title: 'Proceed with evaluating the next document' },
      'Next Document');
    this.nextDocButton_.style.backgroundColor = this.buttonColor_;
    this.nextDocButton_.disabled = true;
    this.nextDocButton_.addEventListener(
      'click', (e) => {
        e.preventDefault();
        this.nextDocument();
    });
    controlPanel.appendChild(
      googdom.createDom(
          'div', null,
          this.prevDocButton_, this.nextDocButton_, progressMessage));

    this.makeEvalPanel();
    if (!config.sqm) {
      // For MQM:
      this.phraseMarker_ = new PhraseMarker(this);
    }

    if (config.sqm) {
      // For SQM, sentence rating UI is always shown (does not require a
      // "severity" button click).
      this.evalPanelBody_.style.display = '';
    }
    this.restoreResults(projectResults);
    this.saveResults();
  }

  /**
   * Returns the approximate number of Lines in descendants with class cls.
   * @param {!Element} elt The parent element,
   * @param {string} cls The class name of descendants.
   * @return {number} The approximate # of lines in descendants with class cls.
   */
  getApproxNumLines(elt, cls) {
    const desc = elt.getElementsByClassName(cls);
    let height = 0;
    for (let i = 0; i < desc.length; i++) {
      height += desc[i].getBoundingClientRect().height;
    }
    // 1.3 is line-height, 150% of 13 is the font-size.
    return Math.ceil(height / (1.3 * 13 * 1.5));
  }

  /**
   * Adjusts the line-height of the smaller column to compensate.
   * @param {!Element} srcTD The source TD cell.
   * @param {!Element} tgtTD The target TD cell.
   */
  adjustHeight(srcTD, tgtTD) {
    if (!srcTD || !tgtTD) {
      return;
    }
    srcTD.style.lineHeight = 1.3;
    tgtTD.style.lineHeight = 1.3;
    const srcLines = this.getApproxNumLines(srcTD, 'source-para');
    const tgtLines = this.getApproxNumLines(tgtTD, 'target-para');
    let perc = 100;
    if (srcLines < tgtLines) {
      perc = Math.floor(100 * tgtLines / srcLines);
    } else {
      perc = Math.floor(100 * srcLines / tgtLines);
    }
    if (perc >= 101) {
      const smaller = (srcLines < tgtLines) ? srcTD : tgtTD;
      smaller.style.lineHeight = 1.3 * perc / 100;
    }
  }

  /**
   * Set the rectangle's coordinates from the box, after scaling.
   */
  setRectStyle(rect, box, scale) {
    rect.style.left = '' + (box.x * scale) + 'px';
    rect.style.top = '' + (box.y * scale) + 'px';
    rect.style.width = '' + (box.w * scale) + 'px';
    rect.style.height = '' + (box.h * scale) + 'px';
  }

  /**
   * Set up zoom-on-hover on the image shown in imgCell.
   */
  setUpImageZooming(imgWrapper, imgCell, zoom, url, w, h, selBox, scale) {
    const zoomW = 600;
    const zoomH = 200;
    const box = {x: 0, y: 0, w: zoomW, h: zoomH};
    this.setRectStyle(zoom, box, 1);

    const zoomImg = googdom.createDom(
      'img', {'class': 'anthea-context-image-zoom-bg', 'src': url,
              'width': w, 'height': h});
    zoom.appendChild(zoomImg);

    const zoomSel = googdom.createDom(
        'div',
        {'class':
             'anthea-context-image-zoom-bg anthea-context-image-selection'});
    this.setRectStyle(zoomSel, selBox, 1);
    zoom.appendChild(zoomSel);

    imgWrapper.addEventListener('mousemove', (e) => {
      zoom.style.display = 'none';
      const bodyRect = document.body.getBoundingClientRect();
      const cellRect = imgCell.getBoundingClientRect();

      const scaledX = (e.pageX - cellRect.left + bodyRect.left) / scale;
      if (scaledX < 0 || scaledX >= w) {
        return;
      }
      const scaledY = (e.pageY - cellRect.top + bodyRect.top) / scale;
      if (scaledY < 0 || scaledY >= h) {
        return;
      }
      zoomImg.style.top = '' + (0 - scaledY) + 'px';
      zoomImg.style.left = '' + (0 - scaledX) + 'px';
      zoomSel.style.top = '' + (selBox.y - scaledY) + 'px';
      zoomSel.style.left = '' + (selBox.x - scaledX) + 'px';

      const wrapperRect = imgWrapper.getBoundingClientRect();
      const x = e.pageX - wrapperRect.left + bodyRect.left;
      const y = e.pageY - wrapperRect.top + bodyRect.top;
      zoom.style.left = '' + x + 'px';
      zoom.style.top = '' + y + 'px';

      zoom.style.display = '';
    });
    imgWrapper.addEventListener('mouseleave', (e) => {
      zoom.style.display = 'none';
    });
  }

  /**
   * Extract page contexts, if provided via an annotation on the first segment.
   */
  extractPageContexts() {
    this.manager_.log(this.manager_.INFO,
                      'Extracting page contexts from annotations');
    for (let i = 0; i < this.docs_.length; i++) {
      const thisDoc = this.docs_[i];
      if (!thisDoc.docsys.annotations ||
          thisDoc.docsys.annotations.length == 0 ||
          !thisDoc.docsys.annotations[0]) {
        this.manager_.log(this.manager_.WARNING,
                          'No annotation (hence no page context) for doc ' + i);
        continue;
      }
      try {
        const pageContext = JSON.parse(thisDoc.docsys.annotations[0]);
        if (!pageContext.source || !pageContext.target) {
          this.manager_.log(
              this.manager_.ERROR,
              'Incomplete page context in the annotation for doc ' + i);
          continue;
        }
        thisDoc.srcContext = pageContext.source;
        thisDoc.tgtContext = pageContext.target;
      } catch (err) {
        this.manager_.log(
            this.manager_.ERROR,
            'Unparseable page context in the annotation for doc ' + i);
        continue;
      }
    }
  }

  /**
   * If the current document has available page context (screenshots of source
   * and translation, and bounding boxes for the text getting evaluated), then
   * show it in the context row.
   */
  showPageContextIfPresent() {
    const doc = this.docs_[this.currDoc_];
    this.contextRow_.innerHTML = '';
    this.contextRow_.style.display = 'none';
    if (!doc.srcContext || !doc.tgtContext) {
      return;
    }

    /**
     * Keep image width down to fit more content vertically. But if
     * the height is very large then use a larger width.
     */
    const width = Math.max(doc.srcContext.h,
                           doc.tgtContext.h) > 2000 ? 450 : 320;

    /**
     * Slightly complex layout, to allow scrolling images vertically, and
     * yet let the zoomed view spill outside. The zoomed view also shows
     * the selection outline.
     * td
     *   anthea-context-image-wrapper
     *     anthea-context-image-port (scrollable)
     *       anthea-context-image-cell
     *         img
     *         anthea-context-image-selection
     *     anthea-context-image-zoom
     *       full-img
     *       full-selection
     */
    const srcImg = googdom.createDom(
        'img', {src: doc.srcContext.url,
                class: 'anthea-context-image', width: width});
    const srcScale = width / doc.srcContext.w;
    const srcSelection = googdom.createDom('div',
                                           'anthea-context-image-selection');
    this.setRectStyle(srcSelection, doc.srcContext.box, srcScale);
    const srcCell = googdom.createDom(
        'div', 'anthea-context-image-cell', srcImg, srcSelection);
    const srcPort = googdom.createDom('div',
                                      'anthea-context-image-port', srcCell);
    const srcZoom = googdom.createDom('div', 'anthea-context-image-zoom');
    srcZoom.style.display = 'none';
    const srcWrapper = googdom.createDom(
        'div', 'anthea-context-image-wrapper', srcPort, srcZoom);
    this.contextRow_.appendChild(googdom.createDom('td', null, srcWrapper));
    this.setUpImageZooming(srcWrapper, srcCell, srcZoom, doc.srcContext.url,
                           doc.srcContext.w, doc.srcContext.h,
                           doc.srcContext.box, srcScale);

    const tgtImg = googdom.createDom(
        'img', {src: doc.tgtContext.url,
                class: 'anthea-context-image', width: width});
    const tgtScale = width / doc.tgtContext.w;
    const tgtSelection = googdom.createDom('div',
                                           'anthea-context-image-selection');
    this.setRectStyle(tgtSelection, doc.tgtContext.box, tgtScale);
    const tgtCell = googdom.createDom(
        'div', 'anthea-context-image-cell', tgtImg, tgtSelection);
    const tgtPort = googdom.createDom('div',
                                      'anthea-context-image-port', tgtCell);
    const tgtZoom = googdom.createDom('div',
                                      'anthea-context-image-zoom');
    tgtZoom.style.display = 'none';
    const tgtWrapper = googdom.createDom(
        'div', 'anthea-context-image-wrapper', tgtPort, tgtZoom);
    this.contextRow_.appendChild(googdom.createDom('td', null, tgtWrapper));
    this.setUpImageZooming(tgtWrapper, tgtCell, tgtZoom, doc.tgtContext.url,
                           doc.tgtContext.w, doc.tgtContext.h,
                           doc.tgtContext.box, tgtScale);

    this.contextRow_.appendChild(googdom.createDom('td',
                                                   'anthea-context-eval-cell'));
    this.contextRow_.style.display = '';

    const sOpt = {block: "center"};
    srcSelection.scrollIntoView(sOpt);
    tgtSelection.scrollIntoView(sOpt);
  }

  /**
   * Sets up the eval. This is the main starting point for the JavaScript code,
   *     and is called when the HTML DOM is loaded.
   *
   * @param {!Element} evalDiv The DIV in which to create the eval.
   * @param {!Object} config The template configuration object.
   * @param {!Array<!Object>} projectData Project data, including src/tgt
   *     sentences. The array also may have srcLang/tgtLang properties.
   * @param {?Array<!Object>} projectResults Previously saved partial results.
   * @param {number=} hotwPercent Percent rate for HOTW testing.
   */
  setUpEval(evalDiv, config, projectData, projectResults, hotwPercent=0) {
    evalDiv.innerHTML = '';

    const instructionsPanel = googdom.createDom('div',
                                                'anthea-mqm-instructions');
    instructionsPanel.id = 'anthea-mqm-instructions-panel';
    evalDiv.append(instructionsPanel);

    this.hotwPercent_ = hotwPercent;

    this.evalCounters_ = {
      numSentenceGroups: 0,
      numPreceding: 0,
      numParagraphBreaks: 0,
      numDocuments: 0,
      numWords: 0,
      numRevisits: 0,
      numFullContextToggles: 0,
    };
    this.lastTimestampMS_ = Date.now();
    this.evalResults_ = [];
    this.sentGroups_ = [];
    this.numTgtWords_ = [];
    this.docs_ = [];
    this.currDoc_ = 0;

    this.contextRow_ = googdom.createDom('tr', 'anthea-context-row');
    this.contextRow_.style.display = 'none';

    this.expanderText_ = googdom.createDom(
      'td', { colspan: 3, class: 'anthea-expander-text' }, 'Expand');
    this.expanderRow_ = googdom.createDom(
      'tr', 'anthea-expander-row', this.expanderText_);
    this.expanderRow_.addEventListener(
      'click', () => { this.toggleExpansion(); });

    const evalHeading = this.READ_ONLY ?
        'Evaluations (view-only)' : 'Evaluations';
    const srcHeading = projectData.srcLang ?
        ('Source (' + projectData.srcLang + ')') : 'Source';
    const tgtHeading = projectData.tgtLang ?
        ('Target (' + projectData.tgtLang + ')') : 'Target';
    const docTextTable = googdom.createDom(
        'table', 'anthea-document-text-table',
        googdom.createDom(
            'tr', null,
            googdom.createDom('td', 'anthea-text-heading', srcHeading),
            googdom.createDom('td', 'anthea-text-heading', tgtHeading),
            googdom.createDom('td', 'anthea-text-heading', evalHeading)),
            this.contextRow_,
            this.expanderRow_);
    evalDiv.appendChild(docTextTable);

    for (let docsys of projectData) {
      const doc = {
        'docsys': docsys,
      };
      this.docs_.push(doc);
      doc.eval = googdom.createDom('div', 'anthea-document-eval-div');
      const docTextSrcRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      const docTextTgtRow = googdom.createDom('td',
                                              'anthea-document-text-cell');
      doc.row = googdom.createDom(
          'tr', null, docTextSrcRow, docTextTgtRow,
          googdom.createDom('td', 'anthea-document-eval-cell', doc.eval));
      if (this.docs_.length > 1) {
        doc.row.style.display = 'none';
      }
      doc.startSG = this.evalResults_.length;
      doc.numSG = 0;

      docTextTable.appendChild(doc.row);

      const srcSentGroups = docsys.srcSentGroups;
      const tgtSentGroups = docsys.tgtSentGroups;
      let srcSpannified = '<p class="anthea-source-para">';
      let tgtSpannified = '<p class="anthea-target-para">';
      for (let i = 0; i < srcSentGroups.length; i++) {
        if (srcSentGroups[i].length == 0) {
          /* New paragraph. */
          srcSpannified = srcSpannified + '</p><p class="anthea-source-para">';
          tgtSpannified = tgtSpannified + '</p><p class="anthea-target-para">';
          this.evalCounters_.numParagraphBreaks++;
          continue;
        }

        const srcWordSpans =
            AntheaEval.spannifyWords(srcSentGroups[i]).spannified;
        this.srcSpansHTML.push(srcWordSpans);

        const tgtSpannifyRet = AntheaEval.spannifyWords(tgtSentGroups[i]);
        this.evalCounters_.numWords += tgtSpannifyRet.numWords;
        this.numTgtWords_.push(tgtSpannifyRet.numWords);
        let tgtWordSpans = tgtSpannifyRet.spannified;
        this.tgtSpansHTML.push(tgtWordSpans);

        let injectedError = '';
        if (!config.sqm && !this.READ_ONLY &&
            this.hotwPercent_ > 0 &&
            i < srcSentGroups.length - 1 &&
            (100 * Math.random()) < this.hotwPercent_) {
          let ret = AntheaEval.injectErrors(tgtSentGroups[i]);
          if (!ret.corrupted) {
            this.tgtRealSpansHTML.push('');
          } else {
            tgtWordSpans = AntheaEval.spannifyWords(ret.corrupted).spannified;
            injectedError = ret.display;
            this.tgtRealSpansHTML.push(tgtWordSpans);
          }
        } else {
          this.tgtRealSpansHTML.push('');
        }

        srcSpannified = srcSpannified +
            "<span class='anthea-source-sent-group'>" +
            srcWordSpans + " </span>";
        tgtSpannified = tgtSpannified +
            "<span class='anthea-target-sent-group'>" +
            tgtWordSpans + " </span>";
        const evalResult = {
          'errors': [],
          'doc': this.docs_.length - 1,
          'visited': false,
          'timestamp': this.lastTimestampMS_,
          'timing': {},
        };
        if (injectedError) {
          evalResult['hotw'] = {
            'timestamp': this.lastTimestampMS_,
            'injected_error': injectedError,
            'done': false,
            'found': false,
          };
        }
        this.evalResults_.push(evalResult);
        doc.numSG++;

        this.sentGroups_.push({
            doc: this.docs_.length - 1, clickListener: null, top: '0', });
      }
      this.evalCounters_.numDocuments++;

      googdom.setInnerHtml(docTextSrcRow, srcSpannified + '</p>');
      googdom.setInnerHtml(docTextTgtRow, tgtSpannified + '</p>');
      this.adjustHeight(docTextSrcRow, docTextTgtRow);
    }

    this.evalCounters_.numSentenceGroups = this.evalResults_.length;

    this.srcSpans = document.getElementsByClassName("anthea-source-sent-group");
    this.tgtSpans = document.getElementsByClassName("anthea-target-sent-group");

    const controlPanel = document.createElement('div');
    controlPanel.id = 'anthea-control-panel';
    evalDiv.append(controlPanel);

    // Compute source_side_ok and source_side_only from subtypes.
    for (let type in config.errors) {
      const errorInfo = config.errors[type];
      if (!errorInfo.subtypes || errorInfo.subtypes.length == 0) {
        if (!errorInfo.source_side_only) errorInfo.source_side_only = false;
        if (!errorInfo.source_side_ok) {
          errorInfo.source_side_ok = errorInfo.source_side_only;
        }
        continue;
      }
      errorInfo.source_side_only = true;
      errorInfo.source_side_ok = false;
      for (let subtype in errorInfo.subtypes) {
        const subtypeInfo = errorInfo.subtypes[subtype];
        if (!subtypeInfo.source_side_only) subtypeInfo.source_side_only = false;
        if (!subtypeInfo.source_side_ok) {
          subtypeInfo.source_side_ok = subtypeInfo.source_side_only;
        }
        errorInfo.source_side_only &&= subtypeInfo.source_side_only;
        errorInfo.source_side_ok ||= subtypeInfo.source_side_ok;
      }
    }

    this.currSentenceGroup = 0;
    this.createUIFromConfig(config, projectResults,
                            instructionsPanel, controlPanel);

    // Extract page contexts if the config expects them.
    if (config.USE_PAGE_CONTEXT) {
      this.extractPageContexts();
    }
    this.showPageContextIfPresent();

    this.recomputeTops();
    this.resizeListener_ = () => { this.recomputeTops(); };
    window.addEventListener('resize', this.resizeListener_);

    this.redrawAllSentenceGroups();
  }
}

/**
 * The PhraseMarker class is used to collect highlighted phrases for the current
 *     sentence group (currSentenceGroup).
 * @final
 */
class PhraseMarker {
  /**
   * @param {!AntheaEval} contextedEval
   */
  constructor(contextedEval) {
    /** @private @const {!AntheaEval} */
    this.contextedEval_ = contextedEval;

    /** @private @const {string} */
    this.DEFAULT_COLOR = 'gainsboro';
    /** @private {string} */
    this.color_ = this.DEFAULT_COLOR;

    /** @private @const {!Object} */
    this.textFroms_ = {
      SOURCE: 0,
      TARGET: 1,
    };
    /** @private {number} */
    this.textFrom_ = this.textFroms_.SOURCE;

    /** @private {number} */
    this.startSpanIndex_ = -1;
    /** @private {number} */
    this.endSpanIndex_ = -1;

    /** @private {!Array<!Element>} Word spans on the source side */
    this.srcWordSpans_ = [];
    /** @private {!Array<!Element>} Word spans on the translation side */
    this.tgtWordSpans_ = [];
    /** @private {!Array<string>} Saved colors of src spans during marking */
    this.srcSpanColors_ = [];
    /** @private {!Array<string>} Saved colors of tgt spans during marking */
    this.tgtSpanColors_ = [];
  }

  /**
   * Resets the word spans in the current sentence groups, getting rid of any
   *     event listeners from spannification done in the previous state. Sets
   *     element class to 'anthea-word-active' or
   *     'anthea-space-active'.
   */
  resetWordSpans() {
    const ce = this.contextedEval_;
    ce.redrawSentenceGroup(ce.currSentenceGroup);

    this.srcWordSpans_ =
      ce.srcSpans[ce.currSentenceGroup].getElementsByTagName('SPAN');
    this.tgtWordSpans_ =
      ce.tgtSpans[ce.currSentenceGroup].getElementsByTagName('SPAN');

    const allowSpaceStart = ce.config.ALLOW_SPANS_STARTING_ON_SPACE || false;

    this.srcSpanColors_ = [];
    const spanClassSuffix =
        (ce.config.MARK_SPAN_FIRST && this.startSpanIndex_ < 0) ? '-begin' : '';
    const suffix = '-active' + spanClassSuffix;
    const spaceClass = 'anthea-space' + suffix;
    const wordClass = 'anthea-word' + suffix;
    for (let x = 0; x < this.srcWordSpans_.length; x++) {
      this.srcWordSpans_[x].className =
          this.srcWordSpans_[x].className + suffix;
      if (allowSpaceStart && this.srcWordSpans_[x].className == spaceClass) {
        this.srcWordSpans_[x].className = wordClass;
      }
      this.srcSpanColors_.push(this.srcWordSpans_[x].style.backgroundColor);
    }
    this.tgtSpanColors_ = [];
    for (let x = 0; x < this.tgtWordSpans_.length; x++) {
      this.tgtWordSpans_[x].className =
          this.tgtWordSpans_[x].className + suffix;
      if (allowSpaceStart && this.tgtWordSpans_[x].className == spaceClass) {
        this.tgtWordSpans_[x].className = wordClass;
      }
      this.tgtSpanColors_.push(this.tgtWordSpans_[x].style.backgroundColor);
    }
  }

  /**
   * Colors the background starting from the span starting at startSpanIndex
   *     and ending at spanIndex (which may be < startSpanIndex_)
   * @param {number} spanIndex
   */
  highlightTo(spanIndex) {
    const spanArray = (this.textFrom_ == this.textFroms_.SOURCE) ?
        this.srcWordSpans_ : this.tgtWordSpans_;
    if (spanIndex >= spanArray.length || spanIndex < 0) {
      return;
    }
    const colorArray = (this.textFrom_ == this.textFroms_.SOURCE) ?
        this.srcSpanColors_ : this.tgtSpanColors_;
    for (let x = 0; x < spanArray.length; x++) {
      const span = spanArray[x];
      if ((x >= this.startSpanIndex_ && x <= spanIndex) ||
          (x <= this.startSpanIndex_ && x >= spanIndex)) {
        span.style.backgroundColor = this.color_;
      } else {
        span.style.backgroundColor = colorArray[x];
      }
    }
  }

  /**
   * Completes the selection of a highlighted phrase, at the span indexed by
   *     spanIndex.
   * @param {number} spanIndex
   */
  pickEnd(spanIndex) {
    if (spanIndex < this.startSpanIndex_) {
      this.endSpanIndex_ = this.startSpanIndex_;
      this.startSpanIndex_ = spanIndex;
    } else {
      this.endSpanIndex_ = spanIndex;
    }
    /* Remove anthea-word listeners. */
    this.resetWordSpans();
    /* But re-do the highlighting. */
    this.highlightTo(this.endSpanIndex_);

    let spanArray = (this.textFrom_ == this.textFroms_.SOURCE) ?
        this.srcWordSpans_ :
        this.tgtWordSpans_;
    let prefix = '';
    for (let x = 0; x < this.startSpanIndex_; x++) {
      prefix = prefix + spanArray[x].innerText;
    }
    let selected = '';
    for (let x = this.startSpanIndex_; x <= this.endSpanIndex_; x++) {
      selected = selected + spanArray[x].innerText;
    }

    const markedPhrase = {
      location:
          ((this.textFrom_ == this.textFroms_.SOURCE) ? 'source' :
                                                        'translation'),
      start: this.startSpanIndex_,
      end: this.endSpanIndex_,
      prefix: prefix,
      selected: selected,
      color: this.color_,
    };
    this.contextedEval_.setMQMPhrase(markedPhrase);
  }

  /**
   * Notes that startSpanIndex for the highlighted phrase is at spanFrom on
   *     the textFrom side (source vs translation), and sets up the UI for
   *     picking the end of the phrase.
   * @param {number} textFrom
   * @param {number} spanIndex
   */
  prepareToPickEnd(textFrom, spanIndex) {
    const ce = this.contextedEval_;
    ce.setStartedMarking();
    ce.showGuidance('Click on the end of the error span');

    this.textFrom_ = textFrom;
    this.startSpanIndex_ = spanIndex;

    /* Remove anthea-word listeners, add new ones. */
    this.resetWordSpans();

    const spanArray = (textFrom == this.textFroms_.SOURCE) ?
        this.srcWordSpans_ : this.tgtWordSpans_;
    const span = spanArray[spanIndex];
    span.style.backgroundColor = this.color_;

    for (let x = 0; x < spanArray.length; x++) {
      spanArray[x].addEventListener(
        'mouseover', () => { this.highlightTo(x); });
      spanArray[x].addEventListener('click', () => { this.pickEnd(x); });
    }
    ce.noteTiming('began-marking-phrase');
  }

  /**
   * Sets state and UI to wait for the start of the highlighted phrase to get
   *     picked.
   */
  prepareToPickStart() {
    const ce = this.contextedEval_;
    this.textFrom_ = this.textFroms_.TARGET;
    this.startSpanIndex_ = -1;
    this.endSpanIndex_ = -1;
    this.resetWordSpans();

    if (ce.config.MARK_SPAN_FIRST) {
      ce.showGuidance('Click on the start of an error span not yet marked');
    } else {
      ce.showGuidance('Click on the start of the error span');
    }
    const cls =
        'anthea-word-active' + (ce.config.MARK_SPAN_FIRST ? '-begin' : '');
    for (let x = 0; x < this.srcWordSpans_.length; x++) {
      if (this.srcWordSpans_[x].className == cls) {
        this.srcWordSpans_[x].addEventListener(
          'click',
          () => { this.prepareToPickEnd(this.textFroms_.SOURCE, x); });
      }
    }
    for (let x = 0; x < this.tgtWordSpans_.length; x++) {
      if (this.tgtWordSpans_[x].className == cls) {
        this.tgtWordSpans_[x].addEventListener(
          'click',
          () => { this.prepareToPickEnd(this.textFroms_.TARGET, x); });
      }
    }
  }

  /**
   * The public entrypoint in the PhraseMarker object. Sets up the UI to
   *     collect a highlighted phrase from currSentenceGroup.
   *     When phrase-marking is done, the contextedEval_ object's
   *     setMQMPhrase() function will get called.
   * @param {string} color The color to use for highlighting.
   */
  getMarkedPhrase(color) {
    this.color_ = color || this.DEFAULT_COLOR;
    this.prepareToPickStart();
  }
}
