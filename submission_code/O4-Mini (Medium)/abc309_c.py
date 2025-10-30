import sys
import threading

def main():
    import sys
    data = sys.stdin.read().split()
    it = iter(data)
    N = int(next(it))
    K = int(next(it))
    meds = []
    total_b = 0
    for _ in range(N):
        a = int(next(it))
        b = int(next(it))
        meds.append((a, b))
        total_b += b

    # If from day 1 already ≤ K, answer is 1
    if total_b <= K:
        print(1)
        return

    # Sort by expiration day a_i ascending
    meds.sort(key=lambda x: x[0])

    # Iterate, dropping off medicines as we pass their last day
    i = 0
    while i < N:
        curr_a = meds[i][0]
        sum_b = 0
        # subtract all medicines that expire on curr_a
        while i < N and meds[i][0] == curr_a:
            sum_b += meds[i][1]
            i += 1
        total_b -= sum_b
        # On day curr_a+1 these are dropped; check if load ≤ K
        if total_b <= K:
            print(curr_a + 1)
            return

    # In case we never hit ≤K earlier, once all are dropped total_b=0 ≤K,
    # so the answer is one day after the last expiration.
    # But the loop above would have caught it, so no need for extra print here.

if __name__ == "__main__":
    main()