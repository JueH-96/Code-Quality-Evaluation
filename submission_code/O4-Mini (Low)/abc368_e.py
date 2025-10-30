import sys
import threading
def main():
    import sys
    input = sys.stdin.readline
    N, M, X1 = map(int, input().split())
    A = [0]*(M+1)
    B = [0]*(M+1)
    S = [0]*(M+1)
    T = [0]*(M+1)
    for i in range(1, M+1):
        a,b,s,t = map(int, input().split())
        A[i],B[i],S[i],T[i] = a,b,s,t

    # We maintain for each city c the current maximum of X[i] + T[i]
    # among all trains i that have arrived at city c with arrival time <= current time.
    mx = [0]*(N+1)

    # Prepare events: (time, type, train_index)
    # type 0 = arrival, type 1 = departure
    events = []
    for i in range(1, M+1):
        events.append((T[i], 0, i))
        events.append((S[i], 1, i))
    # Sort by time, and arrivals (0) before departures (1) at same time
    events.sort(key=lambda x: (x[0], x[1]))

    X = [0]*(M+1)
    for time, typ, i in events:
        if typ == 1:
            # departure of train i
            if i == 1:
                # override with given X1
                X[i] = X1
            else:
                # must satisfy X[i] >= max_over_prev(mx[A[i]] - S[i], 0)
                need = mx[A[i]] - S[i]
                if need < 0:
                    need = 0
                X[i] = need
        else:
            # arrival of train i: update city B[i]
            val = X[i] + T[i]
            c = B[i]
            if val > mx[c]:
                mx[c] = val

    # Print X[2..M]
    out = " ".join(str(X[i]) for i in range(2, M+1))
    print(out)

if __name__ == "__main__":
    main()