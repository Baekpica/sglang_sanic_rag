# LLM 연구 문서를 위한 RAG 시스템

이 프로젝트는 LLM(Large Language Model) 연구 관련 문서들에 대한 질의응답을 제공하는 RAG(Retrieval-Augmented Generation) 기반 시스템입니다.

SGLang 프레임워크를 사용하여 LLM 모델과의 인터페이스를 제공하며, sanic을 사용하여 비동기 웹 서버를 구동하고, ChromaDB를 사용하여 벡터 데이터베이스를 관리합니다.

## 개요

이 시스템은 LLM 연구자들이 PDF와 HTML 문서에서 정보를 쉽게 검색하고 관련 답변을 얻을 수 있도록 설계되었습니다. 문서는 청크로 분할되고 벡터 데이터베이스에 저장되어 효율적인 검색이 가능합니다.

## 주요 기능

- PDF 및 HTML 문서에서 텍스트 추출
- 텍스트 청크 분할 및 임베딩 생성
- 질문과 관련된 문서 콘텐츠 검색
- LLM을 통한 검색된 콘텐츠 기반 응답 생성
- 스트리밍 응답 지원

## 설치 및 실행

### 시스템 요구사항

- NVIDIA GPU 지원 환경 (실험환경: NVIDIA RTX 4090 24GB)
- Docker 설치
- [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) 설치 (GPU 지원 환경에서 필요)
- 지원 환경:
  - Python 3.10
  - PyTorch 2.5
  - CUDA 12.5.1

### Docker 빌드

```bash
# Docker 이미지 빌드
docker build -t sglang-sanic-rag .
```

### 실행 방법

제공된 스크립트를 사용하여 Docker 컨테이너를 실행할 수 있습니다. 스크립트의 옵션을 조정하여 실행할 수 있습니다.:

```bash
./run.sh
```

또는 직접 Docker 명령어로 실행:

```bash
docker run --gpus all ..."
```

*주의: 초기 실행시 LLM 모델 다운로드에 시간이 소요될 수 있습니다.*

## 데이터 준비

`data` 디렉토리에 PDF 또는 HTML 형식의 LLM 연구 문서를 넣으세요. (예제 데이터로는 일부 논문과 블로그 포스팅이 담겨 있습니다.) 예시:
- `data/llm_research_paper_1.pdf`
- `data/llm_research_paper_2.html`
- `data/llm_research_paper_3.pdf`


## API 사용방법

### RAG

```bash
curl -X POST http://localhost:1112/serve \
  -H "Content-Type: application/json" \
  -d '{"question": "LLM의 추론 시간 자기 개선에 대해 설명해주세요.", "stream": false}'
```

## Configuration

| 매개변수            | 설명                        | 기본값                                                        |
|-------------------|---------------------------|-------------------------------------------------------------|
| --model_path      | LLM 모델 경로               | recoilme/recoilme-gemma-2-9B-v0.4                             |
| --data_path       | 문서 데이터 디렉토리         | ./data                                                      |
| --host            | 서버 호스트                  | 0.0.0.0                                                     |
| --port            | 서버 포트                    | 1112                                                        |
| --retriever_model | 임베딩 모델                  | sentence-transformers/static-similarity-mrl-multilingual-v1   |
| --n_results       | 검색할 문서 수               | 2                                                           |
| --max_new_tokens  | 생성할 최대 토큰 수          | 2048                                                        |
| --temperature     | 생성 온도 (높을수록 다양한 결과) | 0.01                                                        |
| --renew_db        | DB 갱신 여부                 | False                                                       |
| --max_chunk_size  | 최대 청크 크기               | 2048                                                        |
| --overlap         | 청크 간 겹침 크기            | 128                                                         |


## 시스템 구조

- 문서 처리: BeautifulSoup(HTML) 및 PyPDF2(PDF)를 사용하여 텍스트 추출
- 임베딩 생성: SentenceTransformer 모델을 사용하여 텍스트 임베딩 생성
- 벡터 저장소: ChromaDB를 사용한 벡터 데이터베이스 관리
- 웹 서버: Sanic 비동기 웹 프레임워크
- LLM 추론: sglang을 통한 LLM 모델과의 인터페이스


## 설계 고려사항

- SGLang: 유사한 문서를 반복적으로 검색하는 RAG 시스템의 특성 상 KV Caching을 통해 고속 추론을 지원하는 SGLang 프레임워크를 LLM 서빙 엔진으로 사용
- Gemma-2-9B: 개발환경(RTX4090)을 기준으로 한국어를 지원하는 모델을 사용하였으며, 성능은 Open Ko-LLM LeaderBoard의 모델 성능를 참고하여 튜닝된 모델을 선택
- ChromaDB: 벡터 데이터베이스로 ChromaDB를 사용하여 벡터 검색을 효율적으로 수행하고 Data path 설정을 통해 document 및 embedding을 저장하며, Renew DB 옵션을 통해 새로운 문서가 추가되었을 때 DB를 갱신하거나, 변화가 없는 경우 기존 DB를 사용할 수 있도록 설계
- Chunking: 문서를 청크로 분할하여 벡터 데이터베이스에 저장하고, Overlap을 통해 중복된 정보를 포함하는 청크를 생성하여 검색 성능을 향상시키고자 했으며, Context로 포함 가능한 n_results 수와 Chunk 크기 등을 사용자가 설정할 수 있도록 설계하여 인프라 환경에 따라 최적화 가능.
- Sanic: SGLang의 기본 API 프레임워크는 FastAPI이나, 비동기 웹 프레임워크인 Sanic을 통해 커스터마이징한 method를 제공하고, 빠른 응답 속도를 제공하며, on-premise 구동환경을 고려하여 asyncio를 통해 단일 워커가 큐에서 Request를 하나씩 처리할 수 있도록 설계 (클라우드 환경에서는 RabbitMQ, SQS, Kafka 등을 사용하여 Scale-out 가능)
- Static Similarity MRL Multilingual(Retriever): 문서 검색을 위한 정적 임베딩 모델로 단일 인스턴스에서의 서빙을 고려하여 CPU Job에서 고속 추론이 가능한 모델을 선택
- Argument parser를 통해 Configuration을 관리하며, 주요 매개변수를 인프라 환경에 따라 변경할 수 있도록 설계

## Limitations & Future Work

- 가변적인 Chat Template을 지원 (AutoTokenizer를 통한 템플릿 생성 후 응답 생성 시 잘못된 응답이 생성되는 경우를 발견하여 Prompt를 hardcoding 해놓은 상태이고, 아직 개선이 필요함. Gemma의 경우 user, model의 상호작용만 존재하나 현 프롬프트에서는 user, model, system의 상호작용이 존재함.)
- 문서 분할: 현재는 문서를 청크로 분할하여 벡터 데이터베이스에 저장하고 있으나, 문서의 특성에 따라 문단 단위로 분할하여 저장하거나, 문서의 특정 부분을 저장하는 방식으로 변경하여 성능 향상이 가능
- RAG 기법 고도화: Contextual Retrieval 등 다양한 RAG 기법을 적용하여 성능 향상 필요
- MSA 아키텍처 적용: 현재는 단일 인스턴스 및 단일 프로세스에서의 서빙만을 고려하여 Sanic 서버 구동을 중심으로 모든 구성요소가 Monolithic하게 동작하도록 설계되어 있으나, MSA 아키텍처를 적용하여 서비스를 분리하고, Scale-out이 가능하도록 설계 필요
