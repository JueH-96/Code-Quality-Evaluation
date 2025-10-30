import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        print('Yes')
        return
    
    n, m = map(int, data[0].split())
    constraints_dict = {}
    
    for i in range(1, m + 1):
        parts = data[i].split()
        x = int(parts[0])
        y = int(parts[1])
        c = parts[2]
        if x not in constraints_dict:
            constraints_dict[x] = {'max_black': 0, 'min_white': 10**10}
        if c == 'B':
            constraints_dict[x]['max_black'] = max(constraints_dict[x]['max_black'], y)
        else:
            constraints_dict[x]['min_white'] = min(constraints_dict[x]['min_white'], y)
    
    rows = []
    for x, d in constraints_dict.items():
        L_val = d['max_black']
        min_white_val = d['min_white']
        if min_white_val == 10**10:
            U_val = n
        else:
            U_val = min_white_val - 1
        if L_val > U_val:
            print('No')
            return
        rows.append((x, L_val, U_val))
    
    rows.sort(key=lambda x: x[0])
    
    last_value = n
    for r, L, U in rows:
        candidate = min(last_value, U)
        if candidate < L:
            print('No')
            return
        last_value = candidate
    
    print('Yes')

if __name__ == "__main__":
    main()