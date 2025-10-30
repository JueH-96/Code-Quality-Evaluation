import sys

def main():
    data = sys.stdin.read().splitlines()
    n = int(data[0])
    strings = data[1].split()
    
    root = {'count': 0, 'children': {}}
    for s in strings:
        node = root
        for c in s:
            if c in node['children']:
                node = node['children'][c]
                node['count'] += 1
            else:
                new_node = {'count': 1, 'children': {}}
                node['children'][c] = new_node
                node = new_node
                
    total_ans = 0
    stack = [root]
    while stack:
        node = stack.pop()
        for child in node['children'].values():
            cnt = child['count']
            total_ans += cnt * (cnt - 1) // 2
            stack.append(child)
            
    print(total_ans)

if __name__ == "__main__":
    main()