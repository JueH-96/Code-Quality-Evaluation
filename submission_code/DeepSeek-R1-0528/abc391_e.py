import sys

def main():
    data = sys.stdin.read().splitlines()
    if not data:
        return
    N = int(data[0].strip())
    s = data[1].strip()
    total_length = 3 ** N
    nodes = []
    for char in s:
        if char == '0':
            nodes.append(('0', 0, 1))
        else:
            nodes.append(('1', 1, 0))
    
    for level in range(1, N + 1):
        new_nodes = []
        for i in range(0, len(nodes), 3):
            n0 = nodes[i]
            n1 = nodes[i + 1]
            n2 = nodes[i + 2]
            count0 = 0
            if n0[0] == '0':
                count0 += 1
            if n1[0] == '0':
                count0 += 1
            if n2[0] == '0':
                count0 += 1
            orig_char = '0' if count0 >= 2 else '1'
            
            a0, b0 = n0[1], n0[2]
            a1, b1 = n1[1], n1[2]
            a2, b2 = n2[1], n2[2]
            
            cost0 = min(
                a0 + a1 + a2,
                a0 + a1 + b2,
                a0 + b1 + a2,
                b0 + a1 + a2
            )
            
            cost1 = min(
                b0 + b1 + b2,
                b0 + b1 + a2,
                b0 + a1 + b2,
                a0 + b1 + b2
            )
            
            new_nodes.append((orig_char, cost0, cost1))
        nodes = new_nodes
    
    orig, cost0, cost1 = nodes[0]
    if orig == '0':
        print(cost1)
    else:
        print(cost0)

if __name__ == '__main__':
    main()