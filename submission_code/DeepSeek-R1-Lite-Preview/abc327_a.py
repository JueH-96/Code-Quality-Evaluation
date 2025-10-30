# Read input N (not used in the logic, but required by the problem)
N = int(input())

# Read the string S
S = input().strip()

# Iterate through pairs of consecutive characters
for c1, c2 in zip(S, S[1:]):
    if {c1, c2} == {'a', 'b'}:
        print("Yes")
        break
else:
    print("No")