def solve():
    n, x = map(int, input().split())
    machines = []
    for _ in range(n):
        machines.append(list(map(int, input().split())))

    def check(capacity):
        total_cost = 0
        for a, p, b, q in machines:
            min_cost_process = float('inf')

            # Iterate through number of S machines
            for num_s in range(capacity // a + 2 if a > 0 else 1):
                if a > 0:
                    production_s = num_s * a
                    if production_s >= capacity:
                        min_cost_process = min(min_cost_process, num_s * p)
                    else:
                        remaining_capacity = capacity - production_s
                        if b > 0:
                            num_t = (remaining_capacity + b - 1) // b
                            min_cost_process = min(min_cost_process, num_s * p + num_t * q)
                elif capacity == 0:
                    min_cost_process = 0
                elif b > 0:
                    num_t = (capacity + b - 1) // b
                    min_cost_process = min(min_cost_process, num_t * q)

            # Iterate through number of T machines
            for num_t in range(capacity // b + 2 if b > 0 else 1):
                if b > 0:
                    production_t = num_t * b
                    if production_t >= capacity:
                        min_cost_process = min(min_cost_process, num_t * q)
                    else:
                        remaining_capacity = capacity - production_t
                        if a > 0:
                            num_s = (remaining_capacity + a - 1) // a
                            min_cost_process = min(min_cost_process, num_s * p + num_t * q)
                elif capacity == 0:
                    min_cost_process = 0
                elif a > 0:
                    num_s = (capacity + a - 1) // a
                    min_cost_process = min(min_cost_process, num_s * p)

            if min_cost_process == float('inf') and capacity > 0:
                return False
            elif capacity == 0:
                min_cost_process = 0

            total_cost += min_cost_process

        return total_cost <= x

    left, right = 0, 2 * 10**9  # Adjust upper bound as needed
    ans = 0
    for _ in range(100): # Binary search iterations
        mid = (left + right) // 2
        if check(mid):
            ans = mid
            left = mid
        else:
            right = mid

        if left == right -1 and check(right):
            ans = right
            break
        elif left == right -1:
            break
        elif left == right:
            break

    print(ans)

solve()