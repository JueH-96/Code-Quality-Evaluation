# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    import math

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    t = int(sys.stdin.readline())
    for _ in range(t):
        S = sys.stdin.readline().strip()
        X = sys.stdin.readline().strip()
        Y = sys.stdin.readline().strip()

        cnt0_X = X.count('0')
        cnt1_X = X.count('1')
        cnt0_Y = Y.count('0')
        cnt1_Y = Y.count('1')
        len_S = len(S)

        delta_len = len_S * (cnt0_X - cnt0_Y)
        delta_cnt_T = cnt1_Y - cnt1_X

        if delta_cnt_T == 0:
            if delta_len == 0:
                len_T = 1  # Arbitrary positive integer
            else:
                print('No')
                continue
        else:
            if delta_len % delta_cnt_T != 0:
                print('No')
                continue
            len_T = delta_len // delta_cnt_T
            if len_T < 0:
                print('No')
                continue

        idx_X = 0
        idx_Y = 0
        i = 0
        j = 0
        n_X = len(X)
        n_Y = len(Y)
        len_S = len(S)
        T_candidate = None
        success = True
        while idx_X < n_X and idx_Y < n_Y:
            c_X = X[idx_X]
            c_Y = Y[idx_Y]
            if c_X == '0' and c_Y == '0':
                # Both are S blocks
                idx_X += 1
                idx_Y += 1
                i += len_S
                j += len_S
            elif c_X == '1' and c_Y == '1':
                # Both are T blocks
                if len_T == 0:
                    idx_X += 1
                    idx_Y += 1
                    continue
                substring_X = ''
                substring_Y = ''
                if len_T > 0:
                    substring_X = ''
                    substring_Y = ''
                    # Due to constraints, i + len_T may exceed limits
                    if i + len_T > i:
                        substring_X = S + ' ' * (i + len_T - len(S))
                        substring_X = substring_X[i:i+len_T]
                    if j + len_T > j:
                        substring_Y = S + ' ' * (j + len_T - len(S))
                        substring_Y = substring_Y[j:j+len_T]
                    else:
                        substring_X = ''
                        substring_Y = ''
                    substring_X = S[i:i+len_T]
                    substring_Y = S[j:j+len_T]
                else:
                    substring_X = ''
                    substring_Y = ''
                if T_candidate is None:
                    T_candidate = substring_X
                else:
                    if substring_X != T_candidate or substring_Y != T_candidate:
                        success = False
                        break
                idx_X += 1
                idx_Y += 1
                i += len_T
                j += len_T
            else:
                # One is S, one is T
                if c_X == '0' and c_Y == '1':
                    # c_X is S, c_Y is T
                    substring_S = S
                    if len_T == 0:
                        T_substring = ''
                    else:
                        T_substring = S[j:j+len_T]
                        if len(T_substring) != len_T:
                            success = False
                            break
                    if T_candidate is None:
                        T_candidate = T_substring
                    else:
                        if T_substring != T_candidate:
                            success = False
                            break
                    if substring_S != T_substring:
                        success = False
                        break
                    idx_X += 1
                    idx_Y += 1
                    i += len_S
                    j += len_T
                elif c_X == '1' and c_Y == '0':
                    # c_X is T, c_Y is S
                    substring_S = S
                    if len_T == 0:
                        T_substring = ''
                    else:
                        T_substring = S[i:i+len_T]
                        if len(T_substring) != len_T:
                            success = False
                            break
                    if T_candidate is None:
                        T_candidate = T_substring
                    else:
                        if T_substring != T_candidate:
                            success = False
                            break
                    if substring_S != T_substring:
                        success = False
                        break
                    idx_X += 1
                    idx_Y += 1
                    i += len_T
                    j += len_S
                else:
                    success = False
                    break
        if success and idx_X == n_X and idx_Y == n_Y:
            print('Yes')
        else:
            print('No')

threading.Thread(target=main).start()