

--- Intro to programming ---
## Day 1 — Lesson 1 (Arithmetic, variables)
- No scars.

## Day 2 — Lesson 2 (Functions)
- Tried to add strings/ints → TypeError. Fixed by casting inputs to int.
- Returned wrong variable name → corrected return value.

## Day 3 — Lesson 3 (Data types)
- Thought vars inside `if` were local → actually global scope in Python.

## Day 4 — Lesson 4 (Conditionals)
- Forgot to cast input → comparison failed. Fixed with int().
- Used wrong variable in return → wrong output. Fixed.

## Day 5 — Lesson 5 (Lists)
- Used `.min`/`.max` as methods → they’re functions. Fixed.
- Misunderstood slices: `[: ]` → until end.
- Negative slice: `[-7:]` → last 7 elements. 
- Added dummy function args → unnecessary if unused.

--- Python Basics ---

## Day 6 
Lesson 1 
- int function rounds down while round function roudns normally
- if you define a variable in the body of an if statement, while it has a global scope if the if statement isnt called it isnt defined : needs to be set with a placeholder value beforehand.

Lesson 2
- Scar: didn’t know how defaults actually worked inside a function
- Fix: defaults are assigned to parameters at definition; if no argument is passed, default is used; if argument is passed, it overrides
- Rule: required first, defaults after (order of defaults changes positional behavior)


## Day 7
Lesson 3
- Got tripped up on double negatives and making strings return true or false
- Fix: convert input to boolean early (`onion = (ask == "yes")`), keep True = wants topping
- Fix: dont design double negatives

Lesson 4
- len(list) is always one more than the value of list.index(final term) due to the index starting at 0
- fix: instead of making both intergers, just use a negative index to find the final term.

## Day 8 
Lesson 4
- Scar: Returned False inside loop → stopped after first check; off-by-one confusion.
- Fix: Only return True inside loop; put return False after loop. Use range(len(xs)-1).
- Insight: range excludes end; len N → N-1 neighbor pairs.
- Rule: Choose one pattern (index or zip); never mix, never early-return False in-loop.

## 9
Scar: Misread .index() — whitespace counts as chars.
Fix: .index() gives first char pos, spaces included.
Insight: Machines count everything.

Scar: Unsure on enumerate() use.
Fix: Use for i, v in enumerate(xs) instead of range(len(xs)).
Rule: Prefer readability over manual indexing.

Scar: Confused .isdigit() vs int().
Fix: .isdigit() checks, int() converts.
Rule: Validate before converting.

----- Numpy ------
 ## 10 array creation,
Scar: Used np.array[[...]] instead of np.array([...]).
Fix: () calls the function; [] defines data.
Rule: () = call, [] = container.

## 11 indexing 
Scar: Expected arr[1] to behave like a string.
Fix: It returns a numeric element, not text.
Rule: Indexing extracts values, not strings.
 
## 12 slicing
Scar: Confused when results became 1-D vs 2-D.
Fix: Integer index collapses; slice keeps the axis.
Rule: Integers remove axes, slices preserve them.

