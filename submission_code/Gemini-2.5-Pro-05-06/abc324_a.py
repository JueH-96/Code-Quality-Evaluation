# YOUR CODE HERE
def solve():
    N = int(input())
    A = list(map(int, input().split()))

    # Convert the list A to a set to get unique elements.
    # If all elements in A are the same, the set will have a size of 1.
    # The problem constraints state 2 <= N, so the list A will never be empty,
    # and set(A) will never be empty.
    if len(set(A)) == 1:
        print("Yes")
    else:
        print("No")

if __name__ == '__main__':
    solve()