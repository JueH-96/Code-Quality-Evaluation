def main():
    import sys
    data = sys.stdin.read().strip().split()
    N = int(data[0])
    takahashi_score = 0
    aoki_score = 0

    idx = 1
    for _ in range(N):
        x = int(data[idx]); y = int(data[idx+1])
        takahashi_score += x
        aoki_score += y
        idx += 2

    if takahashi_score > aoki_score:
        print("Takahashi")
    elif takahashi_score < aoki_score:
        print("Aoki")
    else:
        print("Draw")

# Do not remove the below function call
main()