import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    events = []

    for i in range(1, N + 1):
        T = int(data[2 * i - 1])
        D = int(data[2 * i])
        events.append((T, 'enter', i))
        events.append((T + D, 'leave', i))

    events.sort()

    current_time = 0
    available_products = set()
    printed_count = 0
    last_print_time = -2  # Initialize to -2 to allow printing at time 0

    for time, event_type, product in events:
        if time > current_time:
            current_time = time

        if event_type == 'enter':
            available_products.add(product)
        elif event_type == 'leave':
            available_products.discard(product)

        if available_products and current_time >= last_print_time + 1:
            available_products.pop()
            printed_count += 1
            last_print_time = current_time

    print(printed_count)

if __name__ == "__main__":
    solve()