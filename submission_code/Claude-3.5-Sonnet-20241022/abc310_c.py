N = int(input())
sticks = []
for _ in range(N):
    sticks.append(input())

unique_sticks = set()
for stick in sticks:
    # Add both the original string and its reversal to handle both cases
    unique_sticks.add(min(stick, stick[::-1]))

print(len(unique_sticks))