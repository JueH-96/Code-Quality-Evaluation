n = int(input())
s = input().strip()

beats = {'R': 'S', 'S': 'P', 'P': 'R'}
allowed_moves = {
    'R': ['P', 'R'],
    'P': ['S', 'P'],
    'S': ['R', 'S']
}

prev_allowed = None
prev_values = []

for c in s:
    current_allowed = allowed_moves[c]
    current_values = []
    for x in current_allowed:
        if prev_allowed is None:
            # First move
            current_values.append(1 if beats[x] == c else 0)
        else:
            x0, x1 = prev_allowed
            if x == x0 or x == x1:
                # x is in prev_allowed
                if x == x0:
                    other_val = prev_values[1]
                else:
                    other_val = prev_values[0]
                max_prev = other_val
            else:
                max_prev = max(prev_values[0], prev_values[1])
            # Calculate current value
            current_win = 1 if beats[x] == c else 0
            current_values.append(max_prev + current_win)
    prev_allowed = current_allowed
    prev_values = current_values

print(max(prev_values))