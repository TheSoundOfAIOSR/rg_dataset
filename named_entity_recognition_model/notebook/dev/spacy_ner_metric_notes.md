# PRF Scores

## 1. Information on `Scorer.score()`

NOTE: Skip to the section "3.2 Working of `get_ner_prf()`" if you just want to know how exactly scores are calculated. The TLDR of section 3.1 is that `get_ner_prf()` function is used to calculate the PRF scores.

### 3.1 What happens on `Scorer.score()` call:

* In *scorer.py* `Scorer` class' `score()` function, we are essentially trying to obtain scores of 2 types, first is tokenizer's score, and second is the component's score.
    * The 1st call `self.nlp.tokenizer.score` essentially calls the score function defined in *tokenizer.pyx* file, which calls `score_tokenization` to calculate the accuracy and PRF scores on the tokens, based on the list of `Examples` object.
        * `nlp.tokenizer.score` calls `Scorer.score_tokenization`
    * The 2nd call is done by `component.score(examples, ...)`.
        * The `component` is obtained from `nlp.pipeline`.
        * `nlp` is the loaded nlp object, which is of type `Language` class.
        * `nlp.pipeline` access the pipeline function defined in *language.py* (line 312). This function returns the pipeline, which is essentially a list of tuples made up of `str` and `Doc`.
            ```
            The processing pipeline consisting of (name, component) tuples.

            RETURNS (List[Tuple[str, Callable[[Doc], Doc]]]): The pipeline.
            ```
            The pipeline function returns all those components which are not disabled, and so excludes the disabled ones.
        * In our code:
            ```py
            pipe_exceptions = ["ner"]
            unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
            ```
            In our implementation, we're only enabling the pipepline component "ner".
        * If the pipeline component has any attribute "score", then we can call the score function, which actually calls the `score()` function defined in the `EntityRecognizer` class
        (*`spacy -> pipeline -> ner.pyx`, line no. 191*)
        * This score function essentially returns a `Dict[str, Any]`: The NER precision, recall and f-scores, by calling the function `Scorer.get_ner_prf()` (*line no. 710 in scorer.py*).
            ```py
            # line no. 200 in ner.pyx
            return get_ner_prf(examples)
            ```
* So the TLDR of the above is that spaCy utilizes the `get_ner_prf()` function for calculating the scores on the NER model. 

### 3.2 Working of `get_ner_prf()`

* From the function description: *"Compute micro-PRF and per-entity PRF scores for a sequence of examples."*
* A dictionary consisting of objects of type `PRFScore` class is created, which holds:
    ```py
    tp: int = 0,
    fp: int = 0,
    fn: int = 0,
    ```
* NOTE: This `PRFScore()` class has functions defined as `precision`, `recall` and `fscore` which can calculate and return the respective metrics based on the standard defined formulaes, using the TP, FP and FN variables.

* The gold standard labels and their corresponding entities are extracted from the `Example` object, which was given as parameter to the function.
    ```py
    golds = {(e.label_, e.start, e.end) for e in eg.y.ents}
    ```

* For each type of entity, it looks at each of predicted entity and checks if that entity exists in the gold standard's entity list (aka our training/test data)

* If it does, it counts it as a TP (correctly labelled as [ENTITY]), and then removes this entity from the gold standard list. On the other hand, if the entity does not exist in the gold list, it counts it as FP (incorrectly labelled as [ENTITY]).
    ```py
    key = (pred_ent.label_, indices[0], indices[-1] + 1)
    if key in golds:
        score_per_type[pred_ent.label_].tp += 1
        golds.remove(key)
    else:
        score_per_type[pred_ent.label_].fp += 1
    ```

* After checking all the entities, if there are any entities that have not been crossed off from the gold standard list, it means the model did not tag them (whether correct (TP) or not (FP)) even though they were in the gold standard list, and hence those are falsely labelled as negatives (not labelled at all), aka FN.
    ```py
    for label, start, end in golds:
        score_per_type[label].fn += 1
    ```

* The overall PRF values are calculated by adding the respective TP, FP, FN values (of QLTY and INSTR) together (see the definition of `PRFScore` class for details about the addition). These are [micro-averaged](https://datascience.stackexchange.com/a/24051) scores, wherein essentially, more weightage is given to class with more number of labelled points.
    ```py
    totals = PRFScore()
    for prf in score_per_type.values():
        totals += prf
    ```
    ```py
    # Line no. 43-46 in scorer.py
    def __add__(self, other):
        return PRFScore(
            tp=self.tp + other.tp, fp=self.fp + other.fp, fn=self.fn + other.fn
        )
    ```

* Then it just calculates PRF values using the standard known formulaes by calling the respective defined functions `precision()`, `recall()` and `fscore()`.
    ```py
    if len(totals) > 0:
        return {
            "ents_p": totals.precision,
            "ents_r": totals.recall,
            "ents_f": totals.fscore,
            "ents_per_type": {k: v.to_dict() for k, v in score_per_type.items()},
        }
    ```

---

## 2. Plotting ROC-AUC

* We can make use of  `Scorer.score_cats()`
    * Probably not required for now, as `Scorer.score()` function also outputs the same metrics.
    * TODO: bit more research on this
## 3. Migrating to spaCy v3.x
* spaCy version in use: 3.0.6 (latest)
* [Documentation](https://spacy.io/usage/v3#migrating-training-python) for changing the training style


* If we have our `text` and `annotations` from the minibatch

    ```py
    doc = nlp.make_doc(text)
    example = Example.from_dict(doc, annotations)
    ```

    The idea is that `update()` function is now supposed to be called with a batch of `Example` type objects, rather than `(text, annotation)` tuple.

* The `spacy.gold` module has been renamed to `spacy.training`
    * `GoldParse` has been removed and replaced with `Examples`