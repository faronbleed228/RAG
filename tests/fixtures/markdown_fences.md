# Fenced code variants

```python
# Heading-looking content inside a fence

value = "blank lines inside code should remain code"
```

## Outside the first fence

~~~text
# Heading inside a tilde fence

This is fenced text, even though the delimiter is not a backtick.
~~~

````markdown
## Heading inside a four-backtick fence
```python
value = 42
```
The inner triple backticks do not close the outer fence.
````

## Final real heading

End of document.
