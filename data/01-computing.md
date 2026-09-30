Computing Notes: Small Systems, Clear Interfaces
A computer system becomes easier to understand when its boundaries are visible. A text file is a boundary between a program and persistent storage. A network protocol is a boundary between two machines. A function signature is a boundary between one part of a program and another.

Small systems are not necessarily simple, but they make their complexity easier to inspect. A command-line tool that reads standard input and writes standard output can often be composed with other tools. A compact HTTP service can expose a useful API without requiring a large framework. The important question is not whether a system has many features. It is whether each feature has a clear purpose.

Programs and data
Programs transform data. The transformation may be obvious, such as sorting a list of names, or indirect, such as compiling source code into machine instructions. In both cases, the program needs a representation for its input and a representation for its output.

A useful habit is to write down the data model before writing the implementation. For example, a small task tracker might store tasks like this:


```json
{
  "id": "task-042",
  "title": "Write backup script",
  "done": false,
  "tags": ["operations", "weekend"]
}
```


This does not decide how tasks will be displayed, stored, or synchronized. It does create a stable object around which those decisions can be made.

A simple pipeline
Many useful programs follow a pipeline:

Read input.

Parse it into a structure.

Transform or validate the structure.

Emit output.

Report errors in a form a person can act on.

A log parser is a familiar example. It reads lines, recognizes fields such as timestamps and status codes, counts events, and prints a summary. The parser should not silently discard malformed records. It should say what it expected and where the input differed.

```python
from collections import Counter

counts = Counter()

for line in open("access.log", encoding="utf-8"):
    if " status=" not in line:
        continue
    status = line.split(" status=", 1)[1].split()[0]
    counts[status] += 1

for status, total in sorted(counts.items()):
    print(f"{status}: {total}")
```


The code is small, but it raises design questions. Are log lines trusted? What happens when a file is too large for memory? Should the output be JSON so another program can consume it? A small tool is often the beginning of a larger system.

State and failure
State is information that survives from one operation to the next. A process may keep state in memory, on disk, in a database, or in a message queue. Each location has different failure modes.

Memory is fast but disappears when the process exits. Files are simple and portable but require careful update rules. Databases provide indexing and transactions, but introduce operational requirements. The correct choice depends on the scale and reliability needs of the problem.

When a program writes important state, it should consider interruption. A common pattern is to write a new file, flush it, then rename it into place. On many filesystems, renaming within a directory is more reliable than overwriting a file directly.

Useful constraints
Constraints are a source of design clarity. A program limited to 32 MB of memory must stream data instead of collecting everything in a list. A service that must answer within 50 milliseconds needs bounded work. A tool that must run without network access needs local data and predictable dependencies.

The most durable systems are often built from modest parts: plain files, simple formats, explicit configuration, and observable failure. Fancy technology can be valuable, but it should solve a real constraint rather than hide an unclear design.
