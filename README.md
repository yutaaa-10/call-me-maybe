*This project has been created as part of the 42 curriculum by yukurosa.*

# call-me-maybe


## 概要 (Description)
call-me-maybeは、LLM(Lange Language Model)における
Function Callingの仕組みを学ぶためのプロジェクトです。

Function Callingとは、ユーザーが自然言語で入力した要求をLLMが理解し、
あらかじめ定義された関数の中に適切な関数を選択して、
その関数を呼び出すために必要なパラメータを構造化された形式で出力する仕組みです。

例えば,<br>
"What is the sum of 2 and 3?"<br>
という入力に対して、以下のような出力を生成します。
```
{
	"name": "ft_add_numbers",
	"parameters": {
		"a": 2,
		"b": 3
	}
}
```
<br>
本プロジェクトでは、小規模なLLMである Qwen/Qwen3-0.6B を使用します。
小規模なLLMの出力だけでは常に正しいJSON形式や正しい関数呼び出しを生成できるとは限りません。
そこで、LLMが出力するかくtokenのlogitsに制約を加えるConstrained Decoding(制約付きでコーディング)
を実装し、定義されたJSON構造に従ったFunction Callを生成します。
<br>


## 手順 (Instrunctions)



##　追加の項目 (Additional sectios)

### Algorithm explanation
### Design decisions
### Performance analysis:
### Challenges faced
### Testing strategy
### Example usage
## リソース (Resources)


### AIの使用について

本プロジェクトでは、学習や実装を補助する目的でAIを使用しました。

主に以下の用途で利用しました。
