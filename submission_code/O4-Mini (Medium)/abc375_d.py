import sys
import threading

def main():
    import sys
    S = sys.stdin.readline().strip()
    n = len(S)
    if n < 3:
        print(0)
        return

    # Map characters 'A'..'Z' to 0..25
    A = [ord(c) - ord('A') for c in S]

    # Count of each character to the right of current position
    cnt_right = [0] * 26
    for x in A:
        cnt_right[x] += 1

    cnt_left = [0] * 26
    ans = 0

    # Iterate j from 0 to n-1
    for j in range(n):
        cj = A[j]
        # We are now "at" position j: exclude S[j] from right-count
        cnt_right[cj] -= 1

        # For each letter c, add cnt_left[c] * cnt_right[c]
        # Those represent choices of i<j<k with S[i]=S[k]=c
        # and S[j] arbitrary.
        # Sum over all c gives total palindromic triples centered at j.
        left = cnt_left
        right = cnt_right
        # Unroll small loop for speed
        total = (left[0]*right[0] + left[1]*right[1] + left[2]*right[2] +
                 left[3]*right[3] + left[4]*right[4] + left[5]*right[5] +
                 left[6]*right[6] + left[7]*right[7] + left[8]*right[8] +
                 left[9]*right[9] + left[10]*right[10] + left[11]*right[11] +
                 left[12]*right[12] + left[13]*right[13] + left[14]*right[14] +
                 left[15]*right[15] + left[16]*right[16] + left[17]*right[17] +
                 left[18]*right[18] + left[19]*right[19] + left[20]*right[20] +
                 left[21]*right[21] + left[22]*right[22] + left[23]*right[23] +
                 left[24]*right[24] + left[25]*right[25])
        ans += total

        # Now include S[j] into left-count for future j's
        cnt_left[cj] += 1

    print(ans)


if __name__ == "__main__":
    main()