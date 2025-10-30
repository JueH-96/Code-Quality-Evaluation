Actually, let me step back and think about this more carefully. The issue is that when I'm at position pos, I might not have assigned a value to position a[pos] yet.

Let me try a different approach - process positions in topological order: