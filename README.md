*This project has been created as part of the 42 curriculum by yukurosa.*

# call-me-maybe


## 概要
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


## 手順

### 1. 環境構築

本プロジェクトでは、Pythonのパッケージおよび仮想環境の管理に
`uv` を使用します。
プロジェクトのルートディレクトリで以下を実行し、
必要な依存関係をインストールします。

```bash
uv sync
```

### 2.入力ファイル

プログラムでは、主に以下の2種類のJSONファイルを入力として使用します。

・Function definitions<br>
使用可能な関数の情報を定義します。<br>
各関数には以下の情報が含まれます。<br>

* 関数名
* 関数の説明
* パラメータ名
* パラメータの型
* 戻り値の型

例
```
{
    "name": "fn_add_numbers",
    "description": "Add two numbers",
    "parameters": {
        "a": {
            "type": "number"
        },
        "b": {
            "type": "number"
        }
    },
    "returns": {
        "type": "number"
    }
}
```

・User prompts
LLMに処理させるユーザーの要求をJSON形式で渡します。<br>
例
```
{
    "prompt": "What is the sum of 2 and 3?"
}
```

### 3. 実行

プログラムは以下の形式で実行されます。
```
uv run python -m src \
    --functions_definition <functions_definition.json> \
    --input <input.json> \
    --output <output.json>
```
--functions_definition には関数定義ファイル、<br>
--input にはユーザーpromptの入力ファイル、<br>
--output には生成結果を保存するファイルを指定します。<br>

以下の方法でも実行可能です。
```
make run
```

### 4. 出力

各promptに対して、LLMが選択した関数名と生成したパラメータをJSON形式で出力します。

例
```
{
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
        "a": 2,
        "b": 3
    }
}
```


## 追加の項目

### Algorithm explanation

本プロジェクトでは、LLMが生成する出力を指定されたJSON形式に制御するために、
Constrained Decodeing(制約付きデコーディング)を実装しました。

通常、LLMは次に生成するTokenについて、語彙に存在する全てのTokenの
logitsを出力します。本実装では、その中から現在のJSON構造において、有効なTokenだけを
生成できるように制限します。

処理の大まかな流れは以下の通りです。

1. User promptとFunction definitionsからLLMに渡すコンテキストを作成する
2. コンテキストをToken IDに変換する
3. LLMから次のTokenに対するlogitsを取得する
4. 現在のStateを確認する
5. Stateに応じて、生成可能なTokenを判定する
6. 無効なTokenのlogitsを'-inf'に変更する
7. 有効なTokenの中から次のTokenを選択する
8. 生成したTokenに応じてStateを更新する
9. Function CallのJSONが完成するまで処理を繰り返す

* JSON生成の状態はStateによって管理
```
START
→ FUNCTION_KEY
→ FUNCTION_NAME
→ FUNCTION_SEPARATOR
→ PARAMETERS_KEY
→ PARAMETERS_START
→ PARAMETER_NAME
→ PARAMETER_COLON
→ PARAMETER_VALUE
→ PARAMETER_SEPARATOR / PARAMETERS_END
→ END
```

### Design decisions
実装では、LLMの役割とConstrained Decodingの役割を
できるだけ分離することを意識しました。

LLMには主に以下を判断させます。

- User promptの意味
- 使用するfunction
- Parameterに設定する値

一方でプログラム側では以下を制御

- JSONの構造
- Function definitionsに存在するFunction名
- 選択したFunctionに存在するPramameter名
- Parameterの型に応じたTokenの制約

これにより、特定のpromptに対するFunctionやParameterを
ハードコードせず、異なるpromptやFunction definitionsにも
対応できる設計を目指しました。

また、各処理の責務を明確にするためにコードを複数のファイルに分割しました。

- `models.py`: PydanticモデルとStateの定義
- `preparation.py`: 入力データの読み込み・検証、contextの作成
- `decoder.py`: Stateに応じたConstrained Decodingの制御
- `decoder_mask.py`: logitsに適用する各種mask処理
- `validators.py`: 数値や文字列の検証
- `state_handler.py`: Token生成後のState遷移
- `__main__.py`: プログラム全体の実行制御

入力データの検証にはPydanticを使用し、
Function definitionsやpromptが期待する構造になっていることを確認します。

### Performance analysis:
### Challenges faced
### Testing strategy
### Example usage

以下は、本プログラムを実際に実行する例です。

#### 1. Input files

まず、使用可能なFunctionを定義したJSONファイルを用意します。
`data/input/functions_definition.json`

```json
[
    {
        "name": "fn_add_numbers",
        "description": "Add two numbers",
        "parameters": {
            "a": {
                "type": "number"
            },
            "b": {
                "type": "number"
            }
        },
        "returns": {
            "type": "number"
        }
    }
]
```

次に、Function Callingを行うUser promptを用意します。
`data/input/function_calling_tests.json`
```json
[
    {
        "prompt": "What is the sum of 2 and 3?"
    }
]
```

#### 2. Run the program

プロジェクトのルートディレクトリから、以下のコマンドを実行します。

```bash
uv run python -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json
```

各オプションには以下のファイルを指定します。
- `--functions_definition`: 使用可能なFunctionを定義したJSONファイル
- `--input`: User promptが含まれるJSONファイル
- `--output`: Function Callingの結果を書き込むJSONファイル

プログラムはFunction definitionsとUser promptをLLMに渡し、
Constrained Decodingを使用してFunction Callを生成します。

#### 3. Output
実行後、`--output` で指定したファイルに結果が保存されます。
`data/output/function_calling_results.json`

```json
[
    {
        "prompt": "What is the sum of 2 and 3?",
        "name": "fn_add_numbers",
        "parameters": {
            "a": 2,
            "b": 3
        }
    }
]
```
この例では、
```text
What is the sum of 2 and 3?
```
というUser promptに対して、LLMがFunction definitionsから
```text
fn_add_numbers
```
を選択し、User promptから必要なParameterとして
```text
a = 2
b = 3
```
を生成します。

その結果、自然言語によるUser requestが、
プログラムから利用できる構造化されたFunction Callに変換されます。

#### Another example

例えば、以下のUser promptを入力した場合、

```json
[
    {
        "prompt": "Greet john"
    }
]
```
対応するFunctionがFunction definitionsに存在すれば、
以下のようなFunction Callが生成されます。

```json
[
    {
        "prompt": "Greet john",
        "name": "fn_greet",
        "parameters": {
            "name": "john"
        }
    }
]
```

## Resources

このプロジェクトで必要となる概念や技術を理解するために、以下の資料を参考にしました。

- **LLM（大規模言語モデル）とは？生成AIとの違いや仕組みを解説**
  URL: https://www.nec-solutioninnovators.co.jp/sp/contents/column/20240229_llm.html

- **ChatGPT と Python で学ぶ LLM の ロジット値（対数確率）**
  URL: https://qiita.com/maskot1977/items/8f9fcdc447a96e60aa46

- **制約付きでコーディング**
  URL: https://aipicks.jp/glossary/constrained-decoding

- **PythonでJSONを扱うための方法を解説**
  URL: https://techplay.jp/column/558

- **pydanticによるデータ構造化**
  URL: https://zenn.dev/umi_mori/books/python-programming/viewer/data-structuring-with-pydantic

- **Pythonの次世代パッケージマネージャー「uv」の使い方を徹底解説**
  URL: https://paiza.jp/works/knowledge/article-python-uv-kn


### AIの使用について

このプロジェクトでは、学習および開発のサポートとしてAIを使用しました。
主に以下の用途で使用しました。

- LLM、logits、constrained decodingなど、理解が不十分な概念の壁打ち
- Transformers、PyTorch、Pydanticなど、使用するライブラリや機能の理解
- constrained decodingを実装する際の設計や実装方法の壁打ち
- コードの構造や実装方法についてのレビュー
- READMEなどのドキュメントの構成や文章の改善

AIから得た回答やコードをそのまま使用するのではなく、
内容を確認し、自分で理解・検証した上で実装に反映しました。
また、最終的なプログラムについては自分でテストを行い、
各処理の動作を理解した上で、プロジェクトの要件を満たしていることを確認しました。
