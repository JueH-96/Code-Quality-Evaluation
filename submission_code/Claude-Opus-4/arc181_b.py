Wait, this approach is getting too complex. Let me reconsider.

Actually, a simpler approach: if f(S,T,X) = f(S,T,Y), then for any valid T, the strings must match. The key insight is that we can determine T by looking at the positions where X has 1 and Y has 1.

Let me implement a cleaner solution: