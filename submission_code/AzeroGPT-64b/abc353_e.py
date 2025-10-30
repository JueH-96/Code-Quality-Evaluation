from collections import defaultdict

read = lambda: list(map(str, input().split()))
comp = lambda x, y :1 if x < y else -1 if x > y else 0 
class Node:
    def __init__(self, tot: int = 0, fail = None, pre = 0):
        self.children = defaultdict(Node)
        self.tot = tot
        self.fail = fail
        self.pre = pre

class AhoCorasick:
    def __init__(self):
        self.root = Node()
        self.end = defaultdict(int)
        
    def add(self, w: str):
        cur = self.root
        for c in w:
            cur = cur.children[c]
            cur.tot += 1
        self.end[w] = cur
        
    def build(self):
        que, cur = [], self.root
        
        def f(y, x):
            if not x: return
            if not y.fail and (fa := f(x.fail, x)):
                y.fail = fa
            if (c := comp(y, fa)):
                y.fail = fa.children[c]
            else:
                y.fail = fa
            return y
        
        for k, ch in enumerate(cur.children):
            cur.children[ch] = f(cur.children[ch], cur)
        que.extend(cur.children.values())
        
        while que:
            cur = que.pop(0)
            for k, ch in enumerate(cur.children):
                cur.children[ch] = f(cur.children[ch], cur)
            que.extend(cur.children.values())
            
        cur = self.root
        for suffix in cur.children:
            node = self.end[suffix].fail
            while not node == cur:
                self.end[suffix].pre = max(self.end[suffix].pre, node.pre + len(suffix) - len(node.end))
                node = node.fail
            self.end[suffix].pre = max(self.end[suffix].pre, len(suffix))
                
    def query(self, s: str) -> int:
        res, cnt = 0, defaultdict(int)
        cur = self.root
        for c in s:
            while not (cur and c in cur.children):
                if not cur: break
                cur = cur.fail
            if not cur: break
            cur = cur.children[c]
            res += cur.tot
            res += cur.pre
        return res
    
n, = read()
s, ac, ans = [], AhoCorasick(), []
for i in range(n):
    s.append(input().strip())
    
for st in s:
    ac.add(st)
    
ac.build()
    
for st in s:
    ans.append(ac.query(st))

res = 0
for w in ans:
    res += w
print(res // 2)