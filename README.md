# Gene-Weaver

## Short project overview

GeneWeaver is a solo research and learning project for finding potential SpCas9 target and off-target
locations in a genome.

- **Input:** Genome sequence data and one or more 20-letter guide sequences.
- **Search:** Both DNA strands are checked for a 20-letter region followed by the SpCas9 NGG marker. The
  search allows a chosen number of substitutions in the 20-letter region; candidate windows with unknown
  DNA letters are skipped.
- **Results:** Each candidate includes its guide, chromosome, strand, mismatch count, matching 23-letter
  sequence, and location. Locations start at zero, and the end coordinate is one past the last base.
- **Planned methods:** A CPU search is the baseline. A GPU search is planned and should be considered
  verified only after it matches the CPU results on compatible hardware.
- **Limits:** The first version covers SpCas9 with NGG and substitutions only. It does not cover bulges,
  other Cas enzymes, or predict whether a candidate will actually be cut.

This README describes the development plan; it does not claim that every feature has been implemented.
Mark a day complete only after its deliverable exists and its checks have been carried out.

## The biology in simple terms

- **DNA** is made from four main letters: A, C, G, and T.
- A **guide** is a short sequence that directs the Cas9 enzyme to a similar sequence in DNA.
- Cas9 usually needs a nearby three-letter marker called a **PAM**. This project focuses on the SpCas9 marker
  NGG, where N can be any DNA letter.
- A DNA location that resembles the guide but is not the intended target is called a potential **off-target**.
  Finding these locations helps researchers assess a guide; it does not prove that Cas9 will cut them.

## How to use this 20-day plan

Work through the days in order and keep notes about decisions, results, and problems. Each day has a goal,
practical work, and a completion check. Some goals require an NVIDIA GPU, a real genome, or comparison
software; if a resource is unavailable, record the check as pending. Mark a day complete only when its
deliverable exists and its checks have been carried out.

## Solo development plan

### Phase 1: Plan and prepare

#### Day 1: Define the problem

**Goal:** Understand what the project should find and what it should report.

**Work:** Decide the supported enzyme and marker, strand rules, mismatch limit, input data, and result fields.
Write down what the first version will not support.

**Complete when:** The project scope and expected result are written clearly enough that a small example can be
checked by hand.

#### Day 2: Prepare the project

**Goal:** Make the project organized and repeatable on the development computer.

**Work:** Choose a clear folder structure, record the required tools and libraries, decide where tests and
sample data will live, and make sure changes can be tracked safely.

**Complete when:** The project structure and setup requirements are documented, and a clean environment can
load the project.

#### Day 3: Plan input handling

**Goal:** Specify how genome and guide data will be read and checked.

**Work:** Decide how to handle genome records, wrapped sequence lines, lowercase letters, compressed input,
blank lines, comments, and named guides. Define useful errors for missing or malformed data.

**Complete when:** Valid and invalid input examples are documented, including the expected response to each.

#### Day 4: Define DNA representation

**Goal:** Decide how DNA letters will be represented for comparison.

**Work:** Define one consistent representation for A, C, G, T, and unknown letters. Describe how to convert a
sequence to and from that representation and how to form a reverse complement.

**Complete when:** Hand-checked examples confirm that conversions and reverse complements behave as expected.

#### Day 5: Define target and coordinate rules

**Goal:** Remove ambiguity about what counts as a match.

**Work:** Specify the 20-letter guide region, the adjacent NGG marker on each strand, how unknown letters are
handled, how mismatches are counted, and how locations are reported. Work through one example per strand.

**Complete when:** The expected result for both examples, including strand and location, is recorded.

### Phase 2: Build and verify the search

#### Day 6: Create a simple reference method

**Goal:** Establish a clear, easy-to-check method for finding sites.

**Work:** Describe a straightforward search that examines one genome window at a time, checks the marker,
compares letters, and records matches. Favor clarity over speed.

**Complete when:** A few small hand-made examples can be searched and their expected results explained.

#### Day 7: Develop the CPU search

**Goal:** Make the search practical on a normal processor.

**Work:** Use the agreed sequence representation and target rules. Compare candidate regions with each guide,
count mismatches, and produce consistently ordered results.

**Complete when:** The CPU search agrees with the simple reference method on small examples.

#### Day 8: Check behavior with tests

**Goal:** Make correctness checks repeatable.

**Work:** Prepare small test genomes covering both strands, exact and imperfect matches, unknown letters,
invalid guides, and boundaries between data chunks. Use repeatable test data.

**Complete when:** The checks run consistently and the CPU search agrees with the reference method for all
covered cases.

#### Day 9: Develop the GPU search

**Goal:** Prepare a GPU version of the same search.

**Work:** Plan how independent genome regions will be checked in parallel. Keep the same marker, strand,
mismatch, and result rules as the CPU version. Treat the first GPU result as unverified until Day 12.

**Complete when:** The GPU approach is documented and ready for testing on compatible hardware.

#### Day 10: Support large genomes

**Goal:** Search data that is too large to process all at once.

**Work:** Divide each chromosome into manageable chunks. Define overlap so sites crossing a boundary are
included, and define ownership so a site is reported only once. Keep result ordering deterministic.

**Complete when:** Small-chunk tests show no missing or duplicate boundary sites.

#### Day 11: Make the tool usable

**Goal:** Connect the search to real input and make results easy to use.

**Work:** Decide how a user will provide genome and guide files, choose the mismatch limit and search method,
and choose whether results appear in the terminal or are saved. Prepare a small demonstration dataset.

**Complete when:** The documented workflow takes sample inputs through a complete search and produces an
understandable result table.

### Phase 3: Validate and improve

#### Day 12: Verify the GPU results

**Goal:** Confirm that the GPU and CPU versions find the same sites.

**Work:** On an NVIDIA GPU, run both versions on identical small and sample datasets. Compare complete,
sorted results. Investigate any differences before trusting GPU output.

**Complete when:** The GPU checks pass and the CPU and GPU result tables match exactly on the tested data.
If compatible hardware is unavailable, record this validation as pending.

#### Day 13: Measure performance

**Goal:** Find out which parts of the search take the most time.

**Work:** Choose a repeatable dataset and workload. Measure the CPU and GPU runs and, if possible, use a
profiler to identify where time is spent. Save the hardware and test conditions with the measurements.

**Complete when:** A short record contains the baseline measurements and the main observed bottlenecks.

#### Day 14: Improve performance carefully

**Goal:** Make measured improvements without changing results.

**Work:** Select one bottleneck at a time. Make one focused optimization, rerun the correctness checks, and
compare timings under the same conditions.

**Complete when:** Every retained change preserves the expected results and has a repeatable performance
measurement.

#### Day 15: Explore bulge matching

**Goal:** Plan support for an extra or missing letter in a match.

**Work:** Define guide-side and DNA-side bulges, decide how they should be reported, and prepare independent
examples and tests. Keep this feature separate from ordinary substitution matching.

**Complete when:** The behavior and tests are specified. Treat implementation as future work unless it has
been built and verified.

#### Day 16: Compare with an established tool

**Goal:** Independently compare results with Cas-OFFinder.

**Work:** Use the same genome, guides, and mismatch limit. Align the tools' strand and location conventions
before comparing their result lists. Investigate and document every difference.

**Complete when:** Results agree for the selected cases, or each difference has a documented explanation.
If the comparison cannot be run, record it as pending.

#### Day 17: Benchmark real data

**Goal:** Produce useful, reproducible performance measurements.

**Work:** Select a real genome region and several guide counts and mismatch limits. Record the data source,
hardware, settings, and repeated timings. Do not mix measurements from different conditions.

**Complete when:** The measurements and conditions are saved so another person can understand and repeat them.

#### Day 18: Explain the results

**Goal:** Present performance findings clearly.

**Work:** Summarize the measurements in charts or tables. Explain when the GPU helps, how workload size
affects performance, and what limitations remain. Distinguish measured results from expectations.

**Complete when:** A reader can understand the main findings without guessing how the measurements were made.

### Phase 4: Harden and finish

#### Day 19: Check unusual cases and documentation

**Goal:** Make behavior reliable and explainable.

**Work:** Check empty and very short inputs, unknown-only regions, lowercase sequence, repeated guide names,
large result sets, and small chunks. Review error messages and make sure the README describes verified
behavior rather than planned features.

**Complete when:** The unusual cases have documented outcomes, and unsupported or unverified features are
clearly labeled.

#### Day 20: Repeat the project from a clean start

**Goal:** Confirm another person can understand and reproduce the work.

**Work:** Follow the documentation in a clean environment. Review the project scope, validation evidence,
performance measurements, limitations, and remaining work. Prepare a final summary of the method and results.

**Complete when:** The documented process can be repeated or any remaining blockers are clearly recorded.
