import sys
import threading

def main():
    data = sys.stdin.read().split()
    # First token is N (not strictly needed), second is the string S
    if not data:
        return
    S = data[1]
    N = len(S)
    # left1[i] = number of consecutive '1's immediately before position i
    left1 = [0] * N
    cnt1 = 0
    for i, ch in enumerate(S):
        if ch == '/':
            left1[i] = cnt1
            cnt1 = 0
        elif ch == '1':
            cnt1 += 1
        else:
            # ch == '2'
            cnt1 = 0

    # right2[i] = number of consecutive '2's immediately after position i
    right2 = [0] * N
    cnt2 = 0
    # iterate backwards
    for i in range(N-1, -1, -1):
        ch = S[i]
        if ch == '/':
            right2[i] = cnt2
            cnt2 = 0
        elif ch == '2':
            cnt2 += 1
        else:
            # ch == '1'
            cnt2 = 0

    # Now for each slash position, compute the best 11/22 substring length
    ans = 1  # at least one slash exists, so minimal answer is 1
    for i, ch in enumerate(S):
        if ch == '/':
            k = min(left1[i], right2[i])
            length = 1 + 2 * k
            if length > ans:
                ans = length

    print(ans)

if __name__ == "__main__":
    main()