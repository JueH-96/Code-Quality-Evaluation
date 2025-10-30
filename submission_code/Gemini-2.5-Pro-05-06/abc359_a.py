# YOUR CODE HERE
def solve():
    N = int(input())
    takahashi_count = 0
    for _ in range(N):
        s = input()
        if s == "Takahashi":
            takahashi_count += 1
    print(takahashi_count)

if __name__ == '__main__':
    solve()