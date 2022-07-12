from transformers import pipeline

generator = pipeline("sentiment-analysis")

res = generator(["In this course, we will teach you how to", "Stupid dumb human"])

print(res)


