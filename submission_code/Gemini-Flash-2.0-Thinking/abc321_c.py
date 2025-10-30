from collections import deque

def solve():
    k = int(input())
    queue = deque(range(1, 10))
    count = 0
    last_generated = -1

    all_generated = sorted(list(range(1, 10)))

    while count < k:
        if not queue:
            break

        current = queue.popleft()
        count += 1
        last_generated = current

        last_digit = current % 10
        for digit in range(last_digit):
            next_num = current * 10 + digit
            if next_num not in all_generated:
                queue.append(next_num)
                all_generated.append(next_num)
                all_generated.sort()
                queue = deque(sorted(list(set(queue))))

        if count == k:
            print(last_generated)
            return

solve()