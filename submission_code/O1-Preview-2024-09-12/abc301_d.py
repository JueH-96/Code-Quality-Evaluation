# YOUR CODE HERE

import sys

import threading
def main():
    S = sys.stdin.readline().strip()
    N = int(sys.stdin.readline().strip())
    L = len(S)
    min_possible_from = [0] * (L + 1)
    # Precompute min_possible_from
    for i in range(L - 1, -1, -1):
        ch = S[i]
        bit_index = L - i - 1
        if ch == '1':
            min_possible_from[i] = min_possible_from[i + 1] + (1 << bit_index)
        else:
            min_possible_from[i] = min_possible_from[i + 1]
    if min_possible_from[0] > N:
        print(-1)
        return
    ans = 0
    for i in range(L):
        ch = S[i]
        bit_index = L - i - 1
        if ch == '1':
            ans |= (1 << bit_index)
        elif ch == '0':
            pass  # ans remains the same
        elif ch == '?':
            temp_ans = ans | (1 << bit_index)
            minimal_possible_value = temp_ans + min_possible_from[i + 1]
            if minimal_possible_value <= N:
                ans = temp_ans  # Set bit to '1'
            else:
                pass  # Leave bit as '0'
    if ans > N:
        print(-1)
    else:
        print(ans)
threading.Thread(target=main).start()