import sys

# Read the number of queries
Q = int(sys.stdin.readline())

# Initialize the stack with 100 cards labeled 0
# We use a list as a stack, with the end of the list being the top.
# The initial 100 zeros represent the bottom of the stack that might eventually be reached.
# However, since the stack is guaranteed to not be empty on pop, we only
# need to manage the cards explicitly placed on top.
# A simpler approach is to start with an empty list and only add cards.
# The 100 original cards can be thought of as a reservoir of zeros at the bottom.
# When we pop from the initial state, we get a 0. When we add cards and then pop,
# we pop the added cards. If we pop enough times to exhaust the added cards,
# the next pop effectively comes from the initial 100 zeros.
# Python lists' pop() method removes from the end. If we think of the initial
# 100 zeros as being at the *beginning* of the list, then append adds to the *end*
# (the effective top), and pop from the *end* works correctly.
# The simplest way to represent this is an empty list for the "active" stack,
# and conceptually, the 100 zeros are "below" this. A pop from an empty active stack
# means we've reached one of the initial zeros.

# Let's use a list to represent the cards *added* on top of the initial 100 zeros.
# Popping from this list gives the added cards. If this list becomes empty,
# the next pop effectively comes from the base 100 zeros.
active_stack = []
initial_zeros_count = 100

# We don't actually need to store the 100 zeros. We just need to know how many
# are conceptually left below the cards we add.
# Let's rethink the stack structure. A list where the *end* is the top is standard.
# The initial 100 zeros are at the *bottom*. New cards go on top.
# So, `[0, 0, ..., 0]` (100 times) is the base. Appending adds to the right end.
# Pop removes from the right end. This works perfectly.
# Let's initialize the stack explicitly.

stack = [0] * 100

# Process each query
for _ in range(Q):
    query_line = sys.stdin.readline().split()
    query_type = int(query_line[0])

    if query_type == 1:
        # Type 1: Place card x on top
        x = int(query_line[1])
        stack.append(x)
    elif query_type == 2:
        # Type 2: Remove top card and output
        # Since the stack is guaranteed to have at least one card, pop is safe.
        removed_card_value = stack.pop()
        print(removed_card_value)