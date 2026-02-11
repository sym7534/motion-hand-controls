# Comment Style Guide

A comprehensive guide to writing comments in my personal style. Apply these conventions to any programming language or codebase.

---

## General Tone

- **Casual and conversational** - Write comments like you're talking to yourself or a teammate
- **Lowercase preferred** - Start comments in lowercase unless emphasizing something important
- **Brief and fragmentary** - Complete sentences are optional; short phrases are fine
- **Minimal punctuation** - Skip periods at the end of comments

```
// default values
// subscriptions
// filler
```

---

## Section Headers

Use single-line comments to label logical groups of code. Keep them short (1-3 words).

```
// config + setup
apiKey = process.env.API_KEY
baseUrl = "https://api.example.com"
timeout = 5000

// state
isLoading = false
hasError = false

// handlers + helpers
def onSubmit():
def validateInput():
```

**Common section labels:**
- `// default values`
- `// config` or `// setup`
- `// state`
- `// handlers + helpers`
- `// imports`
- `// constants`
- `// private methods`

Use `+` to combine related items: `// getters + setters`, `// handlers + helpers`, `// read + write`

---

## Inline Explanations

Add brief notes at the end of lines or above them to clarify what code does.

```
timestamp = datetime.now()  # gets current time for logging
config["mode"] = "production"  # default mode

index = (row * cols) + col  // NOT A 2D ARRAY
threshold = 100.0  // arbitrary "occupied" value
```

---

## Import/Include Annotations

When importing modules for specific functionality, note what you're using them from:

```
import math  # for sin + cos
from functools import partial  # for callback binding
#include <functional>  // for std::bind
```

---

## Variable/Index Clarifications

When working with loops or array indices, clarify what variables represent:

```
for i in range(SIZE):  # i = row, j = col
    for j in range(SIZE):
        ...

index = y * cols + x
// row * columns + column

// Y CORRESPONDS WITH ROWS, X CORRESPONDS WITH COLUMNS
```

---

## ALL CAPS for Emphasis

Use ALL CAPS when something is important, needs tuning, or you want to draw attention:

```
// IMPORTANT DATA

// TUNE THESE
updateDistance = 1.5
updateTime = 1000

return data[index] > 20
// TUNE THIS ^^^^
// TO ADJUST THRESHOLD VALUES

// ONLY UPDATE IF VALUE EXISTS (DONT OVERRIDE WITH EMPTY DATA)

// NOT A 2D ARRAY

// FIXED FINALLY
```

---

## Personal Notes and Reactions

It's okay to include casual personal notes, reminders, or reactions:

```
// idk what to set this to
resolution = 0.1

// aagghhgi is this even needed
recalculate()

// mfw implementing a*

// handleClick.. or whatever.. stole the name from some random tutorial

// testing

// works now lol
```

---

## Reference Links

Include URLs to documentation or Stack Overflow when code is based on external resources:

```
// https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API

// https://stackoverflow.com/questions/12345/how-to-do-thing
def convertFormat():
```

---

## Explaining Logic/Purpose

Before a code block, explain what it's doing or why:

```
// keep angle between -pi and pi
while heading > PI:
    heading -= 2 * PI

// if no match found, just use the last item
result = items[-1]

// check if both objects exist before proceeding
if not obj1 or not obj2:
    return

// using bfs to find nearest valid cell
queue = []
```

---

## Algorithm Comments

When implementing algorithms, use comments that mirror pseudocode or textbook descriptions:

```
// OPEN: list of all nodes to be evaluated
openSet = []

// initialize all distances to infinity
distances = [INFINITY] * size

// initialize all "parent" indexes to -1 (no parent)
cameFrom = [-1] * size

// CLOSED: set of nodes already evaluated
closedSet = set()

// current = node in OPEN with the lowest f cost
// remove current from OPEN, add to CLOSED
// if current is the target node, path has been found
// foreach neighbour of the current node
// if neighbour is not traversable or in CLOSED, skip it
// if new path to neighbour is shorter OR neighbour is not in OPEN
// set parent of neighbour to current
// if neighbour is not in OPEN, add neighbour to OPEN
```

---

## Commented-Out Code

Keep alternate implementations or old code as comments when useful for reference:

```
// frameId = data.header.frame_id or "default"
frameId = "world"
```

---

## Function Comments

For functions, a brief note about purpose is enough. No formal docstrings needed:

```
// fetches data from api
def fetchData():

// check for if we're at the goal
def atGoal():

// FROM point a TO point b
def distance(a, b):

// process + transform input
def handleInput():

// state "switching"
def onTimerTick():
```

---

## Grouping Declarations

Use comments to group related variable or member declarations:

```
// network config
apiEndpoint = "..."
timeout = 5000
retryCount = 3

// state tracking
currentValue = None
lastUpdate = None
isProcessing = False

// coordinates
posX = 0.0
posY = 0.0
rotation = 0.0
```

For struct/class members, inline comments work well:

```
gScore = 0.0  // distance from starting node
hScore = 0.0  // distance from end node
fScore = 0.0  // gScore + hScore
```

---

## Multi-line Comments

Use block comments sparingly, mainly for reference material or longer explanations:

```
/* for reference:
The algorithm works by maintaining a priority queue of nodes.
Each iteration, we pop the lowest-cost node and explore its neighbors.
If we find a shorter path to a neighbor, we update its cost and parent.
Continue until we reach the goal or exhaust all possibilities.
*/
```

---

## Language-Specific Comment Syntax

Apply the same style regardless of language:

| Language | Single-line | Block |
|----------|-------------|-------|
| C/C++/Java/JS | `// comment` | `/* comment */` |
| Python | `# comment` | `""" comment """` |
| Ruby | `# comment` | `=begin ... =end` |
| HTML | `<!-- comment -->` | same |
| CSS | `/* comment */` | same |
| SQL | `-- comment` | `/* comment */` |
| Shell/Bash | `# comment` | N/A |

---

## Summary

| Pattern | Example |
|---------|---------|
| Lowercase, casual | `// default values` |
| Section headers | `// handlers + helpers` |
| Use `+` for lists | `// getters + setters` |
| ALL CAPS emphasis | `// TUNE THIS` |
| Personal notes | `// idk what to set this to` |
| Index clarification | `// i = row, j = col` |
| End-of-line notes | `value = 100.0  // arbitrary threshold` |
| Reference links | `// https://stackoverflow.com/...` |
| No periods needed | `// makes a message` |
| Reactions are okay | `// finally works lol` |
