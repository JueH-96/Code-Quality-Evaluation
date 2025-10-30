import sys

def solve():
    n = int(sys.stdin.readline())
    products = []
    for i in range(n):
        t, d = map(int, sys.stdin.readline().split())
        products.append({'id': i+1, 'start_time': t, 'end_time': t + d})
    
    products.sort(key=lambda p: (p['start_time'], p['end_time']))
    
    last_print_time = -1
    printed_count = 0
    
    for product in products:
        start_time = product['start_time']
        end_time = product['end_time']
        possible_print_time = max(start_time, last_print_time + 1)
        if possible_print_time <= end_time:
            print_time = possible_print_time
            last_print_time = print_time
            printed_count += 1
            
    print(printed_count)

if __name__ == '__main__':
    solve()