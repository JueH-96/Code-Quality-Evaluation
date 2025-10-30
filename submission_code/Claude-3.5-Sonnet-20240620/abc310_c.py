# YOUR CODE HERE
def count_unique_sticks(sticks):
    unique = set()
    for stick in sticks:
        unique.add(min(stick, stick[::-1]))
    return len(unique)

N = int(input())
sticks = [input().strip() for _ in range(N)]
print(count_unique_sticks(sticks))