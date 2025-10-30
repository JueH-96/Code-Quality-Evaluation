import bisect

class Fenw:
    def __init__(self, size):
        self.n = size
        self.tree = [0] * (self.n + 1)
    
    def update(self, index, delta):
        if index == 0:
            return
        while index <= self.n:
            self.tree[index] += delta
            index += index & -index
    
    def query(self, index):
        res = 0
        while index:
            res += self.tree[index]
            index -= index & -index
        return res
    
    def range_query(self, l, r):
        if l > r:
            return 0
        res = self.query(r)
        if l - 1 > 0:
            res -= self.query(l - 1)
        return res

class Solution:
    def numberOfAlternatingGroups(self, colors: list, queries: list) -> list:
        n = len(colors)
        tree_count = Fenw(n)
        tree_sum = Fenw(n)
        breaks = []
        break_set = set()
        
        for i in range(n):
            if colors[i] == colors[(i + 1) % n]:
                breaks.append(i)
                break_set.add(i)
        total_breaks = len(breaks)
        breaks.sort()
        
        if total_breaks > 0:
            for i in range(total_breaks):
                cur = breaks[i]
                nxt = breaks[(i + 1) % total_breaks]
                if nxt > cur:
                    arc_len = nxt - cur
                else:
                    arc_len = nxt + n - cur
                tree_count.update(arc_len, 1)
                tree_sum.update(arc_len, arc_len)
        
        res = []
        
        def remove_break(i):
            nonlocal breaks, total_breaks, tree_count, tree_sum, n
            idx = bisect.bisect_left(breaks, i)
            prev_break = breaks[idx - 1] if idx > 0 else breaks[-1]
            next_break = breaks[idx + 1] if idx < len(breaks) - 1 else breaks[0]
            if next_break > i:
                L1 = next_break - i
            else:
                L1 = next_break + n - i
            if i > prev_break:
                L2 = i - prev_break
            else:
                L2 = i + n - prev_break
            if next_break > prev_break:
                newL = next_break - prev_break
            else:
                newL = next_break + n - prev_break
            tree_count.update(L1, -1)
            tree_count.update(L2, -1)
            tree_sum.update(L1, -L1)
            tree_sum.update(L2, -L2)
            tree_count.update(newL, 1)
            tree_sum.update(newL, newL)
            del breaks[idx]
        
        def add_break(i):
            nonlocal breaks, total_breaks, tree_count, tree_sum, n
            if total_breaks == 0:
                arc_len = n
                tree_count.update(arc_len, 1)
                tree_sum.update(arc_len, arc_len)
                breaks.append(i)
                return
            pos = bisect.bisect_left(breaks, i)
            if pos == 0 and pos == len(breaks):
                next_break = breaks[0]
                prev_break = breaks[0]
            else:
                next_break = breaks[pos] if pos < len(breaks) else breaks[0]
                prev_break = breaks[pos - 1] if pos > 0 else breaks[-1]
            if next_break > prev_break:
                oldL = next_break - prev_break
            else:
                oldL = next_break + n - prev_break
            tree_count.update(oldL, -1)
            tree_sum.update(oldL, -oldL)
            if i > prev_break:
                L1 = i - prev_break
            else:
                L1 = i + n - prev_break
            if next_break > i:
                L2 = next_break - i
            else:
                L2 = next_break + n - i
            tree_count.update(L1, 1)
            tree_count.update(L2, 1)
            tree_sum.update(L1, L1)
            tree_sum.update(L2, L2)
            bisect.insort(breaks, i)
        
        for q in queries:
        if q[0] == 1:
                k = q[1]
                if total_breaks == 0:
                    res.append(n)
                else:
                    count = tree_count.range_query(k, n)
                    total = tree_sum.range_query(k, n)
                    ans = total - (k - 1) * count
                    res.append(ans)
            else:
                idx = q[1]
                new_color = q[2]
                old_color = colors[idx]
                if old_color == new_color:
                    continue
                colors[idx] = new_color
                affected = [(idx - 1) % n, idx]
                for i in affected:
                    cur_val = (colors[i] == colors[(i + 1) % n])
                    if i in break_set:
                        if not cur_val:
                            remove_break(i)
                            break_set.discard(i)
                            total_breaks -= 1
                    else:
                        if cur_val:
                            add_break(i)
                            break_set.add(i)
                            total_breaks += 1
        return res