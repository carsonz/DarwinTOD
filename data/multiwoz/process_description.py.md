# MultiWOZ 数据集处理总结

## 任务描述
处理 `/home/zsy/workspace/gitee/IALM/data/multiwoz21/data/dialogues.json` 文件中的 goal description 字段，具体要求：
1. 删除 "task " 后面的编号和冒号
2. 删除每个字母间的 ". "

## 处理结果
- 成功处理了 673 个 description 字段
- 原始文件大小：约 777 万行
- 处理后文件保存为：`/home/zsy/workspace/gitee/IALM/data/multiwoz21/data/dialogues_processed.json`

## 处理前后对比

### 处理前
```
"T. a. s. k.  . 1. 2. 4. 6. 2. :.  . Y. o. u.  . a. r. e.  . l. o. o. k. i. n. g.  . f. o. r.  . a. n.  . e. x. p. e. n. s. i. v. e.  . r. e. s. t. a. u. r. a. n. t.  . a. n. d.  . i. t.  . s. h. o. u. l. d.  . b. e.  . i. n.  . t. h. e.  . e. a. s. t.  . p. a. r. t.  . o. f.  . t. o. w. n. ..  . D. o. n. '. t.  . g. o.  . f. o. r.  . t. h. e.  . f. i. r. s. t.  . v. e. n. u. e.  . t. h. e.  . s. y. s. t. e. m.  . o. f. f. e. r. s.  . y. o. u. ,.  . a. s. k.  . i. f.  . t. h. e. r. e.  . i. s.  . a. n. y. t. h. i. n. g.  . e. l. s. e. ..  . M. a. k. e.  . s. u. r. e.  . y. o. u.  . g. e. t.  . t. h. e.  . a. d. d. r. e. s. s.  . a. n. d.  . p. h. o. n. e.  . n. u. m. b. e. r. ."
```

### 处理后
```
"You are looking for an expensive restaurant and it should be in the east part of town. Don't go for the first venue the system offers you, ask if there is anything else. Make sure you get the address and phone number."
```

## 处理脚本
脚本位置：`/home/zsy/workspace/gitee/IALM/process_description.py`

主要处理步骤：
1. 使用正则表达式删除 "T. a. s. k.  .  数字. :.  " 模式
2. 使用正则表达式删除 "字母. " 模式，只保留字母
3. 使用正则表达式删除 "Task XXXXX:" 格式

## 验证结果
- 原始文件中包含 673 处 "T. a. s. k. " 模式
- 处理后文件中不再包含 "T. a. s. k. " 模式
- 处理后文件中不再包含 "Task XXXXX:" 格式
- description 字段内容已转换为正常可读格式