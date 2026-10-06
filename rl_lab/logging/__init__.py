"""Logging package: concrete BaseLogger observers live here.

Naming note: this subpackage shadows the stdlib `logging` module *inside*
rl_lab only (every file uses absolute imports, verified by the import smoke
tests). Kept because the architecture spec asks for rl_lab/logging/; if it
ever bites a contributor, rename to `observers` in one mechanical commit.
"""
