import sys
import threading

def main():
    import sys

    input = sys.stdin.readline

    T = int(input())
    for _ in range(T):
        N = int(input())
        A = input().strip()
        B = input().strip()

        # positions of 1s in A and in B
        a = [i for i,ch in enumerate(A) if ch=='1']
        b = [i for i,ch in enumerate(B) if ch=='1']

        if not b:
            # B has no 1s: impossible because there is at least one in B by statement
            print(-1)
            continue

        # We need sum A >= sum B
        if len(a) < len(b):
            print(-1)
            continue

        # If B has exactly one 1, answer is max distance from that to any in A
        if len(b)==1:
            target = b[0]
            ans = 0
            for x in a:
                ans = max(ans, abs(x-target))
            print(ans)
            continue

        # Otherwise we have at least 2 ones in B.
        # We will do a linear scan to compute minimal k.
        # We binary‐search k and check if it works by greedy match:
        def can(k):
            # match each B[j] with some A[i] with abs(a[i]-b[j])<=k
            i = 0
            M = len(a)
            for pos in b:
                # advance until a[i] >= pos-k
                while i<M and a[i] < pos-k:
                    i+=1
                if i>=M or a[i] > pos+k:
                    return False
                i+=1
            return True

        lo, hi = 0, N
        while lo<hi:
            mid = (lo+hi)//2
            if can(mid):
                hi = mid
            else:
                lo = mid+1
        print(lo)

if __name__ == "__main__":
    main()