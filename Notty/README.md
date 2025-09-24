

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