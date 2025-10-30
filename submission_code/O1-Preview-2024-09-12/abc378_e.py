# YOUR CODE HERE
import sys
import threading
def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    A = list(map(int, sys.stdin.readline().split()))
    ans = 0
    N_len = len(A)
    for idx, a in enumerate(A):
        i = idx + 1  # 1-based index
        count = i * (N_len - i +1)
        ans += (a % M) * count
    # Note: The problem states that "Note that the outer sum is not taken modulo M."
    # So we don't take the ans modulo M.
    print(ans % (10 ** 9 + 7))  # Assuming large modulus to keep the number manageable.

threading.Thread(target=main).start()