import sys

def main():
    data = sys.stdin.read().split()
    if not data:
        return
    n = int(data[0])
    q = int(data[1])
    parent = list(range(n+1))
    size = [0] * (n+1)
    top = [[] for _ in range(n+1)]
    
    for i in range(1, n+1):
        size[i] = 1
        top[i] = [i]
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    index = 2
    output_lines = []
    
    for _ in range(q):
        t = int(data[index]); index += 1
        if t == 1:
            u = int(data[index]); v = int(data[index+1]); index += 2
            ru = find(u)
            rv = find(v)
            if ru == rv:
                continue
            if size[ru] < size[rv]:
                ru, rv = rv, ru
            parent[rv] = ru
            size[ru] += size[rv]
            list1 = top[ru]
            list2 = top[rv]
            i, j = 0, 0
            merged = []
            while (i < len(list1) or j < len(list2)) and len(merged) < 10:
                if i < len(list1) and (j >= len(list2) or list1[i] >= list2[j]):
                    merged.append(list1[i])
                    i += 1
                elif j < len(list2):
                    merged.append(list2[j])
                    j += 1
            top[ru] = merged
        else:
            v = int(data[index]); k = int(data[index+1]); index += 2
            r = find(v)
            if k <= len(top[r]):
                output_lines.append(str(top[r][k-1]))
            else:
                output_lines.append(str(-1))
    
    print("
".join(output_lines))

if __name__ == '__main__':
    main()