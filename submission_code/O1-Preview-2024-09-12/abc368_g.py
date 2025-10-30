# YOUR CODE HERE
import sys
import threading
import math

def main():
    import sys
    import threading

    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))
    B = list(map(int, sys.stdin.readline().split()))
    Q = int(sys.stdin.readline())

    class PiecewiseLinearFunction:
        def __init__(self, breakpoints, slopes, intercepts):
            self.breakpoints = breakpoints  # List of x where the linear pieces switch
            self.slopes = slopes            # List of slopes for each linear piece
            self.intercepts = intercepts    # List of intercepts for each linear piece

        def compose(self, other):
            # Composes this function with another from the right: f(g(v))
            new_breakpoints = []
            new_slopes = []
            new_intercepts = []

            # Combine the pieces by composing the functions
            for i in range(len(self.breakpoints) - 1):
                for j in range(len(other.breakpoints) - 1):
                    # Compute the composed linear piece over the interval
                    # Get the slopes and intercepts
                    s1 = self.slopes[i]
                    c1 = self.intercepts[i]
                    s2 = other.slopes[j]
                    c2 = other.intercepts[j]

                    # The composed function over this interval is:
                    # f(g(v)) = s2 * (s1 * v + c1) + c2 = (s2 * s1) * v + (s2 * c1 + c2)
                    new_slope = s2 * s1
                    new_intercept = s2 * c1 + c2

                    # Determine the interval of v where this expression is valid
                    v_start = max(self.breakpoints[i], other.breakpoints[j])
                    v_end = min(self.breakpoints[i+1], other.breakpoints[j+1])

                    if v_start < v_end:
                        new_breakpoints.append(v_start)
                        new_slopes.append(new_slope)
                        new_intercepts.append(new_intercept)

            # Add the last breakpoint
            new_breakpoints.append(float('inf'))

            return PiecewiseLinearFunction(new_breakpoints, new_slopes, new_intercepts)

        def evaluate(self, v):
            # Evaluate the piecewise function at value v
            idx = 0
            for i in range(len(self.breakpoints) -1):
                if self.breakpoints[i] <= v < self.breakpoints[i+1]:
                    idx = i
                    break
            else:
                idx = len(self.breakpoints) - 2

            return self.slopes[idx] * v + self.intercepts[idx]

    # Build the segment tree
    class SegmentTreeNode:
        def __init__(self, l, r):
            self.l = l  # Left index (inclusive)
            self.r = r  # Right index (exclusive)
            self.left = None
            self.right = None
            self.func = None  # PiecewiseLinearFunction representing the interval

    def build(node):
        if node.l +1 == node.r:
            # Leaf node
            i = node.l
            A_i = A[i]
            B_i = B[i]
            if B_i ==1:
                # Only Option1 is better
                slopes = [1]
                intercepts = [A_i]
                breakpoints = [float('-inf'), float('inf')]
            else:
                T_i = A_i / (B_i -1)
                slopes = [1, B_i]
                intercepts = [A_i, 0]
                breakpoints = [float('-inf'), T_i, float('inf')]

            node.func = PiecewiseLinearFunction(breakpoints, slopes, intercepts)
        else:
            m = (node.l + node.r) //2
            node.left = SegmentTreeNode(node.l, m)
            node.right = SegmentTreeNode(m, node.r)
            build(node.left)
            build(node.right)
            node.func = compose_functions(node.left.func, node.right.func)

    def compose_functions(f, g):
        # Compose f and g: f(g(v))
        new_breakpoints = []
        new_slopes = []
        new_intercepts = []

        i = j = 0
        f_breakpoints = f.breakpoints
        f_slopes = f.slopes
        f_intercepts = f.intercepts

        g_breakpoints = g.breakpoints
        g_slopes = g.slopes
        g_intercepts = g.intercepts

        # Since the functions are piecewise linear, and each has up to 2 pieces,
        # their composition can have up to 4 pieces.

        # Build all combinations
        for i in range(len(f.breakpoints) -1):
            for j in range(len(g.breakpoints) -1):
                # Compute the composed linear piece over the interval
                s1 = f.slopes[i]
                c1 = f.intercepts[i]
                s2 = g.slopes[j]
                c2 = g.intercepts[j]
                new_slope = s1 * s2
                new_intercept = s1 * c2 + c1

                v_start = max(f.breakpoints[i], g.breakpoints[j])
                v_end = min(f.breakpoints[i+1], g.breakpoints[j+1])

                if v_start < v_end:
                    new_breakpoints.append(v_start)
                    new_slopes.append(new_slope)
                    new_intercepts.append(new_intercept)

        # Sort by breakpoints
        combined = sorted(zip(new_breakpoints, new_slopes, new_intercepts), key=lambda x: x[0])
        final_breakpoints = []
        final_slopes = []
        final_intercepts = []
        last_bp = None
        for bp, s, c in combined:
            if bp != last_bp:
                final_breakpoints.append(bp)
                final_slopes.append(s)
                final_intercepts.append(c)
                last_bp = bp
            else:
                # Keep the one with maximum value
                idx = len(final_slopes) -1
                if s * bp + c > final_slopes[idx] * bp + final_intercepts[idx]:
                    final_slopes[idx] = s
                    final_intercepts[idx] = c

        final_breakpoints.append(float('inf'))
        return PiecewiseLinearFunction(final_breakpoints, final_slopes, final_intercepts)

    def query(node, l, r):
        if node.l >= r or node.r <= l:
            # No overlap
            return None
        if node.l >= l and node.r <= r:
            # Total overlap
            return node.func
        # Partial overlap
        left_func = query(node.left, l, r)
        right_func = query(node.right, l, r)
        if left_func is None:
            return right_func
        elif right_func is None:
            return left_func
        else:
            return compose_functions(left_func, right_func)

    def update(node, idx):
        if node.l +1 == node.r:
            # Leaf node
            i = node.l
            A_i = A[i]
            B_i = B[i]
            if B_i ==1:
                # Only Option1 is better
                slopes = [1]
                intercepts = [A_i]
                breakpoints = [float('-inf'), float('inf')]
            else:
                T_i = A_i / (B_i -1)
                slopes = [1, B_i]
                intercepts = [A_i, 0]
                breakpoints = [float('-inf'), T_i, float('inf')]

            node.func = PiecewiseLinearFunction(breakpoints, slopes, intercepts)
        else:
            if idx < node.left.r:
                update(node.left, idx)
            else:
                update(node.right, idx)
            node.func = compose_functions(node.left.func, node.right.func)

    root = SegmentTreeNode(0,N)
    build(root)

    ans = []
    for _ in range(Q):
        query_line = sys.stdin.readline().strip()
        if not query_line:
            # Read until we get a non-empty line
            query_line = sys.stdin.readline().strip()
        tokens = query_line.strip().split()
        if tokens[0] == '1':
            _, i, x = tokens
            i = int(i) -1
            x = int(x)
            A[i] = x
            update(root, i)
        elif tokens[0] == '2':
            _, i, x = tokens
            i = int(i) -1
            x = int(x)
            B[i] = x
            update(root, i)
        elif tokens[0] == '3':
            _, l, r = tokens
            l = int(l) -1
            r = int(r)
            func = query(root, l, r)
            # Evaluate function starting from v=0
            v = 0
            for i in range(len(func.breakpoints)-1):
                v = func.slopes[i]* v + func.intercepts[i]
            ans.append(int(v))
    for v in ans:
        print(v)

threading.Thread(target=main).start()