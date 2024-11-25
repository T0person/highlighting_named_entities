import pandas as pd
from datasets import Dataset
# from transformers import GPT2LMHeadModel, GPT2TokenizerFast, GPT2Config
# from transformers import get_linear_schedule_with_warmup
import tensorflow as tf
# import torch
# from torch.optim import AdamW
# from torch.utils.data import Dataset, DataLoader
# from torch.utils.data import random_split, RandomSampler, SequentialSampler
from transformers import create_optimizer, TFAutoModelForQuestionAnswering, DefaultDataCollator

df = pd.read_csv('test_items.csv', names=['id', 'item', 'text'])

print(df.info())

description = df.apply(lambda x: x["item"].lower().split()[0] in x["text"].lower(), axis=1)

df = pd.concat([df,description],axis=1,names=['description'])
df = df.rename(columns={0:'description'})


mentions = df[df['description']==True]
primary = df[df['description']==False]

context = df.apply(lambda x: f"{x['item']} : {x['text']}", axis=1)
context = context.rename('context')

end_context = df.apply(lambda x: len(x['item'].strip()), axis=1)
end_context = end_context.rename('end_context')

start_context = df.apply(lambda x: 0, axis=1)
start_context = start_context.rename('start_context')

total_df = pd.concat([context,start_context, end_context],axis=1)

total_df = pd.concat([total_df,df['item'], df['text']],axis=1)
print(total_df)


from transformers import AutoTokenizer
# Токенизатор
tokenizer = AutoTokenizer.from_pretrained("ai-forever/ruBert-base")

def create_dataset(total_df, tokenizer):

    # Беру строки
    questions = [q.strip() for q in total_df["text"]]
    context = [q.strip() for q in total_df["context"]]
    answers = [q.strip() for q in total_df["item"]]
    start = total_df["start_context"].tolist()
    end = total_df["end_context"].tolist()


    inputs = tokenizer(
        questions,
        context,
        max_length=1300,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
        )

    offset_mapping = inputs.pop("offset_mapping")

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = start[i]
        end_char = end[i]-1
        sequence_ids = inputs.sequence_ids(i)

        # Ищу начало и конец контекста
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # Если ответ не полностью соответствует контексту, отмечаю его (0, 0).
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # В противном случае это начальная и конечная позиции токена.
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    # total_df["answer_start"] = start_positions
    # total_df["answer_end"] = end_positions

    # import pandas as pd
    # from datasets import Dataset
    data = {'input_ids': inputs['input_ids'],
            'attention_mask': inputs['attention_mask'],
            'start_positions':start_positions,
            'end_positions': end_positions,
        }
    df = pd.DataFrame(data)
    # df.to_csv('encoding_train.csv',index=False)
    train = Dataset.from_pandas(df)

    return train

ds = create_dataset(total_df, tokenizer)
ds = ds.train_test_split(test_size=0.1)


batch_size = 3
num_epochs = 2
total_train_steps = (len(ds['train']) // batch_size) * num_epochs
optimizer, schedule = create_optimizer(
    init_lr=2e-5,
    num_warmup_steps=0,
    num_train_steps=total_train_steps,
)

data_collator = DefaultDataCollator(return_tensors="tf") # Сборщик данных

model = TFAutoModelForQuestionAnswering.from_pretrained("ai-forever/ruBert-base", from_pt=True)



tf_train_set = model.prepare_tf_dataset(
    ds['train'],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

tf_validation_set = model.prepare_tf_dataset(
    ds['test'],
    shuffle=False,
    batch_size=batch_size,
    collate_fn=data_collator,
)



model.compile(optimizer=optimizer)
model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=2)