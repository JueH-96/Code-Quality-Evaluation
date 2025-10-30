import sys
import threading

def main():
    input = sys.stdin.readline
    T = int(input())
    out = []
    for _ in range(T):
        R, B = map(int, input().split())
        n = R + B

        # Impossible cases:
        #   single piece (1,0) or (0,1), or (1,1).
        if (R, B) in [(1,0),(0,1),(1,1)]:
            out.append("No")
            continue

        # Otherwise we can do it:
        out.append("Yes")

        # Case 1: B == 0, R>=2 --> pure red cycle
        if B == 0:
            if R == 2:
                out.append("R 1 1")
                out.append("R 1 2")
            else:
                # Draw a 2 x ceil(R/2) rectangle
                k = (R + 1)//2
                path = []
                # top row left->right
                for c in range(1, k+1):
                    path.append((1,c))
                # bottom row right->left
                for c in range(k, 0, -1):
                    path.append((2,c))
                # take first R of them
                for i in range(R):
                    r,c = path[i]
                    out.append(f"R {r} {c}")
            continue

        # Case 2: R == 0, B>=2 --> pure blue cycle
        if R == 0:
            if B == 2:
                # just two diagonal neighbors
                out.append("B 1 1")
                out.append("B 2 2")
            else:
                # do a zig-zag diagonal strip of length B
                pts = []
                x = 1
                for i in range(B):
                    # alternate (x,x) then (x+1,x+1)
                    pts.append((x, x + (i%2)))
                    if i%2:
                        x += 1
                for (r,c) in pts:
                    out.append(f"B {r} {c}")
            continue

        # Case 3: mixed R>=2, B>=2
        #
        # We'll build an explicit cycle of length n = R+B
        # whose edges: exactly R are orthogonal steps, B are diagonal steps.
        # Then label each node by the type of outgoing step.
        #
        # Strategy: place points around a wide spiral:
        L = max(2, (n+1)//2)
        # Build a rectangular loop of perimeter 2*L + 2*2 - 4 = 2L
        # with exactly n points on it.  We'll record each step whether
        # it's orthogonal (O) or diagonal (D).  We'll stub out a sequence:
        seq = []
        # walk top edge left->right (orth steps)
        for c in range(1, L):
            seq.append((1,c, 'O'))
        # walk down diagonal corner (diagonal)
        seq.append((1,L,'D'))
        seq.append((2,L,'O'))  # orth
        # walk bottom edge right->left (orth)
        for c in range(L,2,-1):
            seq.append((2,c,'O'))
        # walk diagonal to top-left (diagonal)
        seq.append((2,2,'D'))
        # close the loop by returning to (1,1) via orth
        seq.append((1,1,'O'))

        # now seq is about 2L+2 steps; we only need the first n
        seq = seq[:n]
        # Make sure we got enough
        if len(seq) < n:
            # pad with trivial orthogonal back-and-forth
            r,c = seq[-1][0], seq[-1][1]
            while len(seq)<n:
                seq.append((r,c,'O'))

        # Now output them, labeling each node by its outgoing step
        # 'O' -> place an R (red) there; 'D' -> place a B.
        for (r,c,typ) in seq:
            if typ=='O':
                out.append(f"R {r} {c}")
            else:
                out.append(f"B {r} {c}")

    sys.stdout.write("
".join(out))

if __name__ == "__main__":
    main()