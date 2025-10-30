import sys

def solve():
    n = int(sys.stdin.readline())
    products = []
    for i in range(n):
        t_d = list(map(int, sys.stdin.readline().split()))
        products.append({'id': i+1, 'start_time': t_d[0], 'duration': t_d[1], 'end_time': t_d[0] + t_d[1]})
    
    products.sort(key=lambda p: (p['end_time'], p['start_time']))
    
    last_print_time = -1
    printed_count = 0
    
    for product in products:
        possible_print_time = max(product['start_time'], last_print_time + 1)
        if possible_print_time <= product['end_time']:
            last_print_time = possible_print_time
            printed_count += 1
            
    print(printed_count)

if __name__ == '__main__':
    solve()