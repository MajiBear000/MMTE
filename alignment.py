import jieba

en = 'The invaders spread their language all over the country.'
text = '入侵者在全国各地传播他们的语言。'
s = jieba.lcut(text)
zh = ' '.join(s)

a = jieba.lcut(en)
a = [i for i in a if not i==' ']
en = ' '.join(a)

f = open('en.txt', 'w', encoding='utf-8')
f.write(en)
f.close()