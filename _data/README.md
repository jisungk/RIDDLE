### Data file format
`dummy.txt` is a space-delimited plaintext data file containing randomly generated data. Each line represents the codified medical history of an individual, and obeys the following format:

`[age] [race/ethnicity] [gender] [first ICD9 code]:[age at event] [second ICD9 code]:[age at event] ...`

For example, `27 W F 410.0:18 413.0:25 540.1:26` represents a White, female individual who is 27 years old and has a history of "Acute myocardial infarction of anterolateral wall" (ICD9 = 410.0) at age 18, "Angina decubitus" (ICD9 = 413.0) at age 25, and "Acute appendicitis with peritoneal abscess" (ICD9 = 540.1) at age 26.

### Miscellaneous

`phewas_codes.txt` is a tab-delimited plaintext file providing descriptions of ICD9 codes. It may be helping for fixing typos in ICD9 codes and for interpreting results. It was generously provided by Professor Josh C. Denny of Vanderbilt University Medical Center. 