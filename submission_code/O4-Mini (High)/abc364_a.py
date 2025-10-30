import sys

def main():
    input = sys.stdin.readline
    N = int(input())
    prev_sweet = False
    for i in range(N):
        s = input().strip()
        if s == "sweet":
            if prev_sweet:
                # He feels sick on the second sweet in a row.
                # If it's the last dish, it's OK, else he can't finish.
                if i == N - 1:
                    print("Yes")
                else:
                    print("No")
                return
            prev_sweet = True
        else:
            prev_sweet = False
    # No consecutive sweets or breaks only at the end
    print("Yes")

main()