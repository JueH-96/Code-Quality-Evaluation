# Read the input values
N, T, A = map(int, input().split())

# Check if either candidate's votes exceed half of the total votes (rounded down)
if T > N // 2 or A > N // 2:
    print("Yes")
else:
    print("No")