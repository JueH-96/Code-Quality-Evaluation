N, Q = map(int, input().split())
T = list(map(int, input().split()))

# Initially all holes have teeth
teeth = [True] * (N+1)  # 1-based indexing

# Process each treatment
for t in T:
    teeth[t] = not teeth[t]  # Flip state of tooth at hole t

# Count remaining teeth
result = sum(teeth[1:])
print(result)