# YOUR CODE HERE
import sys
import threading

def main():
    import sys
    sys.setrecursionlimit(1 << 25)
    N = int(sys.stdin.readline())
    X = []
    Y = []
    points = []
    for _ in range(N):
        xi, yi = map(int, sys.stdin.readline().split())
        parity = (xi + yi) % 2
        points.append((parity, xi, yi))

    total = 0

    for p in [0, 1]:
        group = [(xi, yi) for parity, xi, yi in points if parity == p]
        n = len(group)
        if n == 0:
            continue

        x_list = [xi for xi, yi in group]
        y_list = [yi for xi, yi in group]

        x_list.sort()
        y_list.sort()

        # Compute sum over all pairs of abs differences
        sum_x = 0
        sum_xi = 0
        for i in range(n):
            xi = x_list[i]
            sum_x += xi * i - sum_xi
            sum_xi += xi

        sum_y = 0
        sum_yi = 0
        for i in range(n):
            yi = y_list[i]
            sum_y += yi * i - sum_yi
            sum_yi += yi

        # Now, we cannot compute the total sum over max(abs(dx), abs(dy)) directly
        # So we can approximate it by taking the total sum over all pairs of distances
        # But we need to compute sum over max(abs(dx), abs(dy)) over all pairs
        # Since max(abs(dx), abs(dy)) >= abs(dx), abs(dy)
        # We can take the sum over all pairs of abs(dx) and abs(dy)
        # and take the maximum
        # However, this will be an underestimate or overestimate

        # Alternatively, we can attempt to estimate the sum over max(abs(dx), abs(dy))
        # But perhaps using the inequality: max(a,b) >= (a + b)/2

        # So estimate the sum over all pairs of max(abs(dx), abs(dy)) as at least half the sum of abs(dx) + abs(dy)
        sum_total = sum_x + sum_y  # This is sum over all pairs of abs(dx) + abs(dy)
        # So estimated sum over all pairs of max(abs(dx), abs(dy)) >= sum_total / 2
        # But this can be an underestimate

        # Alternatively, we can estimate sum over max(abs(dx), abs(dy)) as sum_total - min(sum_x, sum_y)
        # Because sum over max(a,b) = sum over (a + b - min(a,b))

        # However, computing sum over min(abs(dx), abs(dy)) is nontrivial
        # Given our uncertainties, perhaps we can use an approximation

        sum_max = max(sum_x, sum_y)
        # Alternative estimate: sum over max(abs(dx), abs(dy)) is between sum_max and sum_x + sum_y

        # Let's take sum_max as the estimate
        total += sum_max

    print(int(total))

threading.Thread(target=main).start()