import sys

def solve():
    N, M = map(int, sys.stdin.readline().split())

    # has_taro_been_named[i] will be True if family i has already had its earliest male child named Taro.
    # Initialize all to False. We use N+1 size to allow 1-based indexing for family IDs.
    has_taro_been_named = [False] * (N + 1)

    for _ in range(M):
        A, B = sys.stdin.readline().split()
        family_id = int(A)
        gender = B

        # A baby is named Taro if they are male AND no male has been named Taro in their family yet.
        if gender == 'M' and not has_taro_been_named[family_id]:
            sys.stdout.write("Yes
")
            has_taro_been_named[family_id] = True  # Mark that this family has now had its Taro
        else:
            sys.stdout.write("No
")

# Call the solve function to run the program
if __name__ == '__main__':
    solve()