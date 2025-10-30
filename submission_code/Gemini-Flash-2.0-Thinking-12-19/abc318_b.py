def solve():
    n = int(input())
    rectangles = []
    for _ in range(n):
        a, b, c, d = map(int, input().split())
        rectangles.append({'a': a, 'b': b, 'c': c, 'd': d})
    
    total_area = 0
    for x in range(100):
        for y in range(100):
            is_covered = False
            for rect in rectangles:
                if max(x, rect['a']) <= min(x + 1, rect['b']) and max(y, rect['c']) <= min(y + 1, rect['d']):
                    is_covered = True
                    break
            if is_covered:
                total_area += 1
                
    print(total_area)

if __name__ == '__main__':
    solve()