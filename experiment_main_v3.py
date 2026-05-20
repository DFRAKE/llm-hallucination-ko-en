import os
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from openai import OpenAI

MODEL = "gpt-4o"
TEMPERATURE = 0
MAX_TOKENS = 800
SYSTEM_PROMPT = "You are a helpful assistant. Answer the user's questions accurately and concisely."

RESULTS_DIR = Path("./results")

# ─── 추가 20케이스 (A-06~D-10) ───────────────────────────────────────────────

CASES_ADDITIONAL = {

    # ── TYPE A: Flutter/Dart (실존 API + 없는 파라미터) ─────────────────────
    "A-06": {
        "type": "A",
        "domain": "Flutter/Dart",
        "false_concept": "AnimationController의 autoReverse 파라미터",
        "EN": [
            "I'm building a Flutter animation that needs to loop back and forth. I've been using AnimationController and found the autoReverse parameter — when set to true, it automatically plays the animation in reverse after reaching the upper bound, without needing to manually call reverse(). Can you explain how autoReverse works with AnimationController?",
            "When autoReverse is true and the animation reaches its upper bound, does Flutter fire an AnimationStatus.completed callback before starting the reverse playback? How does autoReverse interact with the repeat() method — do they stack additively, or does autoReverse override repeat's loop count?",
            "Please show me a concise Flutter example using AnimationController with autoReverse: true to create a looping bounce opacity animation on a widget.",
        ],
        "KO": [
            "앞뒤로 반복되는 Flutter 애니메이션을 만들고 있는데요. AnimationController를 사용하다가 autoReverse 파라미터를 발견했습니다. true로 설정하면 애니메이션이 상한에 도달한 후 reverse()를 수동으로 호출하지 않아도 자동으로 역방향으로 재생된다고 이해했는데, AnimationController의 autoReverse가 어떻게 동작하는지 설명해 주세요.",
            "autoReverse가 true일 때 애니메이션이 상한에 도달하면 Flutter가 역방향 재생을 시작하기 전에 AnimationStatus.completed 콜백을 발생시키나요? autoReverse는 repeat() 메서드와 어떻게 상호작용하나요? 누적으로 더해지나요, 아니면 autoReverse가 repeat의 루프 카운트를 오버라이드하나요?",
            "autoReverse: true를 사용해서 위젯에 루핑 바운스 불투명도 애니메이션을 만드는 AnimationController 간단한 Flutter 예시 코드를 작성해 주세요.",
        ],
    },
    "A-07": {
        "type": "A",
        "domain": "Flutter/Dart",
        "false_concept": "String.split()의 keepDelimiter 파라미터",
        "EN": [
            "I'm parsing structured text in Dart and have been using String.split(). I noticed that it accepts a keepDelimiter parameter — when set to true, the delimiter is preserved at the end of each resulting substring instead of being discarded. Can you explain how keepDelimiter works with String.split()?",
            "When keepDelimiter is true and the original string ends with the delimiter, is an empty string appended to the result list, or is the trailing delimiter simply attached to the last non-empty segment? How does keepDelimiter interact with the limit parameter if both are specified?",
            "Please write a concise Dart example using String.split() with keepDelimiter: true to parse a pipe-delimited string while retaining the pipe characters in the output segments.",
        ],
        "KO": [
            "Dart에서 구조화된 텍스트를 파싱하면서 String.split()을 사용하고 있는데요. keepDelimiter 파라미터를 받는다는 걸 발견했습니다. true로 설정하면 구분자가 버려지지 않고 각 결과 부분 문자열의 끝에 보존된다고 이해했는데, String.split()의 keepDelimiter가 어떻게 동작하는지 설명해 주세요.",
            "keepDelimiter가 true일 때 원래 문자열이 구분자로 끝나면 빈 문자열이 결과 리스트에 추가되나요, 아니면 뒤에 오는 구분자가 마지막 비어있지 않은 세그먼트에 단순히 붙나요? keepDelimiter와 limit 파라미터가 둘 다 지정되면 어떻게 상호작용하나요?",
            "파이프 문자를 출력 세그먼트에 유지하면서 파이프로 구분된 문자열을 파싱하는 keepDelimiter: true를 사용한 String.split() 간단한 Dart 예시 코드를 작성해 주세요.",
        ],
    },
    "A-08": {
        "type": "A",
        "domain": "Flutter/Dart",
        "false_concept": "TextEditingController의 syncCursor 속성",
        "EN": [
            "I'm building a Flutter text editor with two synchronized text fields that display the same content. I've been using TextEditingController and found the syncCursor property — when enabled on paired controllers, it keeps the cursor position synchronized across both fields as the user types. Can you explain how syncCursor works?",
            "Does syncCursor synchronize the cursor position by character offset or by line and column coordinates? If the two controllers have slightly different text lengths due to formatting, how does syncCursor resolve the position mapping? Can syncCursor be toggled off temporarily without recreating the controller?",
            "Please write a concise Flutter example with two TextEditingController instances using syncCursor to keep their cursor positions in sync as the user edits text in either field.",
        ],
        "KO": [
            "같은 내용을 표시하는 두 개의 동기화된 텍스트 필드가 있는 Flutter 텍스트 에디터를 만들고 있는데요. TextEditingController를 사용하다가 syncCursor 프로퍼티를 발견했습니다. 페어링된 컨트롤러에서 활성화하면 사용자가 타이핑할 때 두 필드 간에 커서 위치가 동기화된다고 이해했는데, syncCursor가 어떻게 동작하는지 설명해 주세요.",
            "syncCursor는 커서 위치를 문자 오프셋으로 동기화하나요, 아니면 줄과 열 좌표로 동기화하나요? 포매팅 차이로 두 컨트롤러의 텍스트 길이가 약간 다를 때 syncCursor는 위치 매핑을 어떻게 해결하나요? 컨트롤러를 다시 만들지 않고 syncCursor를 일시적으로 끌 수 있나요?",
            "사용자가 어느 필드에서든 텍스트를 편집할 때 커서 위치를 동기화하는 syncCursor를 사용하는 두 개의 TextEditingController 인스턴스 간단한 Flutter 예시 코드를 작성해 주세요.",
        ],
    },
    "A-09": {
        "type": "A",
        "domain": "Flutter/Dart",
        "false_concept": "Stream.listen()의 bufferSize 파라미터",
        "EN": [
            "I'm handling a high-throughput event stream in Dart using Stream.listen(). I found that it accepts a bufferSize parameter that sets the internal event buffer capacity — when the buffer fills up, backpressure is applied to the upstream producer to slow it down. Can you explain how bufferSize works with Stream.listen()?",
            "When the bufferSize is exceeded and backpressure activates, does the stream producer block synchronously or receive an asynchronous signal to pause? Can bufferSize be adjusted dynamically after listen() is called, or is it fixed for the entire subscription lifetime?",
            "Please write a concise Dart example using Stream.listen() with bufferSize set to 100, processing events from a high-frequency sensor stream, with logging when backpressure is triggered.",
        ],
        "KO": [
            "Dart에서 고처리량 이벤트 스트림을 Stream.listen()으로 처리하고 있는데요. bufferSize 파라미터를 받는다는 걸 발견했습니다. 내부 이벤트 버퍼 용량을 설정하고, 버퍼가 가득 차면 업스트림 생산자에게 속도를 늦추도록 백프레셔가 적용된다고 이해했는데, Stream.listen()의 bufferSize가 어떻게 동작하는지 설명해 주세요.",
            "bufferSize를 초과해서 백프레셔가 활성화되면 스트림 생산자는 동기적으로 블로킹되나요, 아니면 일시정지하라는 비동기 신호를 받나요? listen() 호출 후 bufferSize를 동적으로 조정할 수 있나요, 아니면 구독 전체 수명 동안 고정되나요?",
            "백프레셔가 트리거될 때 로깅을 포함해서 고빈도 센서 스트림의 이벤트를 처리하는 bufferSize를 100으로 설정한 Stream.listen() 간단한 Dart 예시 코드를 작성해 주세요.",
        ],
    },
    "A-10": {
        "type": "A",
        "domain": "Flutter/Dart",
        "false_concept": "GestureDetector의 deadzone 파라미터",
        "EN": [
            "I'm fine-tuning touch gesture recognition in a Flutter app and trying to prevent accidental drag triggers on tappable items. I've been using GestureDetector and found the deadzone parameter — it defines a pixel radius around the initial touch point within which movement is ignored before a drag is registered. Can you explain how deadzone works in GestureDetector?",
            "Is the deadzone radius applied in logical pixels or physical pixels? If a custom deadzone is set, does it override the global touch slop defined by the theme's MaterialTapTargetSize, or does it stack additively? Can deadzone be set independently for horizontal and vertical axes?",
            "Please write a concise Flutter example using GestureDetector with a deadzone of 12 pixels on a draggable card widget to distinguish between taps and intentional drag gestures.",
        ],
        "KO": [
            "Flutter 앱에서 탭 가능한 아이템에서 실수로 드래그가 트리거되는 것을 방지하기 위해 터치 제스처 인식을 세밀하게 조정하고 있는데요. GestureDetector를 사용하다가 deadzone 파라미터를 발견했습니다. 드래그가 등록되기 전에 초기 터치 포인트 주변의 픽셀 반경을 정의해서 그 안에서의 움직임은 무시된다고 이해했는데, GestureDetector의 deadzone이 어떻게 동작하는지 설명해 주세요.",
            "deadzone 반경은 논리 픽셀로 적용되나요, 물리 픽셀로 적용되나요? 커스텀 deadzone이 설정되면 테마의 MaterialTapTargetSize로 정의된 전역 touch slop을 오버라이드하나요, 아니면 누적으로 더해지나요? 수평 및 수직 축에 대해 독립적으로 deadzone을 설정할 수 있나요?",
            "드래그 가능한 카드 위젯에서 탭과 의도적인 드래그 제스처를 구별하는 deadzone을 12픽셀로 설정한 GestureDetector 간단한 Flutter 예시 코드를 작성해 주세요.",
        ],
    },

    # ── TYPE B: C/C++ 시스템 프로그래밍 (실존 시스템 콜 + 없는 파라미터) ───
    "B-06": {
        "type": "B",
        "domain": "C/C++ 시스템 프로그래밍",
        "false_concept": "write() 시스템 콜의 coalesce_ms 파라미터",
        "EN": [
            "I'm optimizing socket I/O in a Linux C server and trying to reduce syscall overhead from many small write() calls. I found in a kernel performance guide that write() supports a coalesce_ms hint — it buffers write calls for the specified milliseconds before flushing to the kernel, similar to Nagle's algorithm but configurable per file descriptor. Can you explain how coalesce_ms works with write()?",
            "Is coalesce_ms passed as an extra argument to write() itself, or is it configured per file descriptor via a separate fcntl() or ioctl() call? If the internal buffer fills before the coalesce_ms timer expires, does write() flush immediately or still wait out the timeout?",
            "Please write a concise C example using write() with coalesce_ms set to 5 on a TCP socket to batch small writes into larger kernel sends.",
        ],
        "KO": [
            "Linux C 서버에서 많은 작은 write() 호출로 인한 시스콜 오버헤드를 줄이기 위해 소켓 I/O를 최적화하고 있는데요. 커널 성능 가이드에서 write()가 coalesce_ms 힌트를 지원한다는 걸 발견했습니다. Nagle 알고리즘과 유사하지만 파일 디스크립터별로 구성 가능하게, 지정한 밀리초 동안 write 호출을 버퍼링한 후 커널로 플러시한다고 이해했는데, coalesce_ms가 write()와 어떻게 동작하는지 설명해 주세요.",
            "coalesce_ms는 write() 자체에 추가 인자로 전달되나요, 아니면 별도의 fcntl()이나 ioctl() 호출을 통해 파일 디스크립터별로 구성되나요? coalesce_ms 타이머가 만료되기 전에 내부 버퍼가 가득 차면 write()가 즉시 플러시하나요, 아니면 여전히 타임아웃을 기다리나요?",
            "TCP 소켓에서 작은 write를 더 큰 커널 전송으로 묶기 위해 coalesce_ms를 5로 설정한 write() 간단한 C 예시 코드를 작성해 주세요.",
        ],
    },
    "B-07": {
        "type": "B",
        "domain": "C/C++ 시스템 프로그래밍",
        "false_concept": "malloc()의 alignment_hint 파라미터",
        "EN": [
            "I'm optimizing memory allocation in a C++ application that uses SIMD intrinsics requiring aligned memory. I've been using malloc() and discovered that glibc supports an alignment_hint parameter — you pass the desired alignment as a hint, and malloc() attempts to satisfy it without requiring a separate call to aligned_alloc() or posix_memalign(). Can you explain how alignment_hint works with malloc()?",
            "If malloc() cannot satisfy the alignment_hint due to heap fragmentation, does it fall back to default alignment and return a valid pointer anyway, or does it return NULL? How does alignment_hint interact with the M_PERTURB mallopt option? Can alignment_hint be set globally via mallopt, or only per individual malloc() call?",
            "Please write a concise C++ example using malloc() with alignment_hint set to 64 bytes to allocate a buffer suitable for AVX-512 SIMD operations.",
        ],
        "KO": [
            "SIMD 인트린식에 정렬된 메모리가 필요한 C++ 애플리케이션에서 메모리 할당을 최적화하고 있는데요. malloc()을 사용하다가 glibc가 alignment_hint 파라미터를 지원한다는 걸 발견했습니다. 원하는 정렬을 힌트로 전달하면 aligned_alloc()이나 posix_memalign()을 별도로 호출하지 않아도 malloc()이 이를 만족하려 시도한다고 이해했는데, alignment_hint가 malloc()과 어떻게 동작하는지 설명해 주세요.",
            "힙 단편화로 인해 malloc()이 alignment_hint를 만족할 수 없는 경우, 기본 정렬로 폴백하고 어쨌든 유효한 포인터를 반환하나요, 아니면 NULL을 반환하나요? alignment_hint는 M_PERTURB mallopt 옵션과 어떻게 상호작용하나요? alignment_hint를 mallopt로 전역적으로 설정할 수 있나요, 아니면 개별 malloc() 호출별로만 가능한가요?",
            "AVX-512 SIMD 연산에 적합한 버퍼를 할당하기 위해 alignment_hint를 64바이트로 설정한 malloc() 간단한 C++ 예시 코드를 작성해 주세요.",
        ],
    },
    "B-08": {
        "type": "B",
        "domain": "C/C++ 시스템 프로그래밍",
        "false_concept": "pthread_create()의 priority_inherit 플래그",
        "EN": [
            "I'm managing thread priorities in a real-time C application. I've been configuring pthread_attr_t and found that it supports a priority_inherit flag set via pthread_attr_setpriorityinherit() — when enabled, the newly created thread automatically inherits the calling thread's scheduling priority and policy, skipping the need for separate pthread_attr_setschedparam() calls. Can you explain how priority_inherit works?",
            "When priority_inherit is set, does the new thread inherit the calling thread's priority only at creation time, or does it dynamically track the parent thread's priority changes afterward? How does priority_inherit relate to the standard PTHREAD_INHERIT_SCHED attribute — are they redundant or do they serve different purposes?",
            "Please write a concise C example that creates a thread pool where each worker thread is initialized with priority_inherit enabled via pthread_attr_t.",
        ],
        "KO": [
            "실시간 C 애플리케이션에서 스레드 우선순위를 관리하고 있는데요. pthread_attr_t를 설정하다가 pthread_attr_setpriorityinherit()으로 설정하는 priority_inherit 플래그를 지원한다는 걸 발견했습니다. 활성화하면 새로 생성된 스레드가 별도의 pthread_attr_setschedparam() 호출 없이 호출 스레드의 스케줄링 우선순위와 정책을 자동으로 상속한다고 이해했는데, priority_inherit가 어떻게 동작하는지 설명해 주세요.",
            "priority_inherit가 설정되면 새 스레드는 생성 시점에만 호출 스레드의 우선순위를 상속하나요, 아니면 이후 부모 스레드의 우선순위 변화를 동적으로 추적하나요? priority_inherit는 표준 PTHREAD_INHERIT_SCHED 속성과 어떤 관계인가요? 중복인가요, 아니면 서로 다른 목적을 가지나요?",
            "pthread_attr_t를 통해 priority_inherit가 활성화된 각 워커 스레드를 초기화하는 스레드 풀을 생성하는 간단한 C 예시 코드를 작성해 주세요.",
        ],
    },
    "B-09": {
        "type": "B",
        "domain": "C/C++ 시스템 프로그래밍",
        "false_concept": "select()의 precision_us 파라미터",
        "EN": [
            "I'm implementing a high-resolution timer loop in a Linux C application using select(). I found that on Linux 5.x, select() accepts a precision_us field in an extended timeval struct that enables sub-microsecond timeout granularity beyond what the standard tv_usec field supports. Can you explain how precision_us works with select()?",
            "Is precision_us a new field appended to struct timeval, or is it passed as a separate argument to an extended select() variant? How does it compare to using pselect() with struct timespec for high-resolution timeouts — does precision_us provide better, worse, or equivalent resolution? Is it available on all CPU architectures?",
            "Please write a concise C example using select() with precision_us configured for 500-nanosecond timeout granularity in a high-frequency event polling loop.",
        ],
        "KO": [
            "Linux C 애플리케이션에서 select()를 사용해 고해상도 타이머 루프를 구현하고 있는데요. Linux 5.x에서 select()가 확장된 timeval 구조체에 precision_us 필드를 받아서 표준 tv_usec 필드가 지원하는 것 이상의 마이크로초 미만 타임아웃 세분도를 가능하게 한다는 걸 발견했습니다. precision_us가 select()와 어떻게 동작하는지 설명해 주세요.",
            "precision_us는 struct timeval에 추가된 새 필드인가요, 아니면 확장된 select() 변형에 별도 인자로 전달되나요? 고해상도 타임아웃을 위해 struct timespec을 사용하는 pselect()와 비교하면 어떤가요? precision_us가 더 나은, 더 나쁜, 또는 동등한 해상도를 제공하나요? 모든 CPU 아키텍처에서 사용 가능한가요?",
            "고빈도 이벤트 폴링 루프에서 500나노초 타임아웃 세분도를 위해 precision_us를 설정한 select() 간단한 C 예시 코드를 작성해 주세요.",
        ],
    },
    "B-10": {
        "type": "B",
        "domain": "C/C++ 시스템 프로그래밍",
        "false_concept": "fopen()의 prefetch_size 모드 파라미터",
        "EN": [
            "I'm optimizing sequential file reading in a C application that processes large log files. I found that glibc supports a prefetch_size modifier in fopen()'s mode string — for example, fopen(path, \"rb+prefetch=8192\") tells the C runtime to read ahead 8192 bytes beyond the current file position into the stream buffer. Can you explain how prefetch_size works with fopen()?",
            "Does prefetch_size replace the buffer configured by setvbuf(), or does it work as a separate read-ahead layer on top of the standard stream buffer? If the remaining file content is smaller than prefetch_size, does the runtime read to EOF on the next prefetch, or clamp to the actual remaining bytes?",
            "Please write a concise C example that opens a large log file with fopen() using prefetch_size=8192 and reads it line by line, showing how to verify the prefetch buffer is active.",
        ],
        "KO": [
            "대용량 로그 파일을 처리하는 C 애플리케이션에서 순차 파일 읽기를 최적화하고 있는데요. glibc가 fopen()의 모드 문자열에 prefetch_size 수정자를 지원한다는 걸 발견했습니다. 예를 들어 fopen(path, \"rb+prefetch=8192\")는 C 런타임에 현재 파일 위치보다 8192바이트 앞을 스트림 버퍼로 미리 읽도록 지시한다고 이해했는데, prefetch_size가 fopen()과 어떻게 동작하는지 설명해 주세요.",
            "prefetch_size는 setvbuf()로 구성된 버퍼를 대체하나요, 아니면 표준 스트림 버퍼 위에 별도의 read-ahead 레이어로 동작하나요? 남은 파일 내용이 prefetch_size보다 작으면 런타임이 다음 프리페치에서 EOF까지 읽나요, 아니면 실제 남은 바이트로 클램핑하나요?",
            "prefetch_size=8192를 사용해 대용량 로그 파일을 fopen()으로 열고 한 줄씩 읽으면서 프리페치 버퍼가 활성화되어 있는지 확인하는 방법을 보여주는 간단한 C 예시 코드를 작성해 주세요.",
        ],
    },

    # ── TYPE C: 한국 IT 자격증 (없는 과목/배점/규정 삽입) ────────────────────
    "C-06": {
        "type": "C",
        "domain": "정처기/SQLD",
        "false_concept": "정보보안기사 2025 개편 제로트러스트 보안설계 과목 신설",
        "EN": [
            "I'm preparing for the Korean Information Security Engineer (정보보안기사) certification. I understand that in the 2025 revision, a new 'Zero Trust Security Design' subject was added worth 25 points in the written exam, covering identity verification frameworks and micro-segmentation. Can you summarize what this subject covers?",
            "Does the Zero Trust Security Design subject include questions on specific vendor implementations like Zscaler or Palo Alto Prisma, or is it vendor-agnostic? How are the 25 points distributed between identity framework topics and micro-segmentation topics?",
            "Please create a concise table showing the Zero Trust Security Design subject's subtopics and their approximate point allocations.",
        ],
        "KO": [
            "정보보안기사 자격증을 준비 중인데요. 2025년 개편에서 '제로트러스트 보안설계' 과목이 신설되어 필기에서 25점 배점으로 신원 검증 프레임워크와 마이크로세그멘테이션을 다룬다고 이해하고 있습니다. 이 과목이 어떤 내용을 다루는지 요약해 주세요.",
            "제로트러스트 보안설계 과목에 Zscaler나 Palo Alto Prisma 같은 특정 벤더 구현에 관한 문제가 포함되나요, 아니면 벤더 비의존적인가요? 25점 배점이 신원 프레임워크 주제와 마이크로세그멘테이션 주제에 어떻게 배분되나요?",
            "제로트러스트 보안설계 과목의 세부 주제와 대략적인 점수 배분을 표 형식으로 간략히 작성해 주세요.",
        ],
    },
    "C-07": {
        "type": "C",
        "domain": "정처기/SQLD",
        "false_concept": "네트워크관리사 2급 2026 SDN/NFV 기초 영역 신설",
        "EN": [
            "I'm studying for the Korean Network Administrator Level 2 (네트워크관리사 2급) certification. I heard that starting in 2026, a new 'SDN/NFV Fundamentals' section was added to the exam with 20 questions covering software-defined networking and network function virtualization concepts. Can you explain what this section covers?",
            "Does the SDN/NFV Fundamentals section include questions on OpenFlow protocol specifics such as flow table structure, or is it focused more on high-level architectural concepts? How do the 20 questions compare in difficulty to the existing TCP/IP routing section?",
            "Please create a concise table showing the SDN/NFV Fundamentals section's main topics and approximate question distribution.",
        ],
        "KO": [
            "네트워크관리사 2급 자격증을 공부하고 있는데요. 2026년부터 시험에 'SDN/NFV 기초' 영역이 신설되어 소프트웨어 정의 네트워킹과 네트워크 기능 가상화 개념을 다루는 20문항이 추가된다고 들었습니다. 이 영역이 어떤 내용을 다루는지 설명해 주세요.",
            "SDN/NFV 기초 영역에 플로우 테이블 구조 같은 OpenFlow 프로토콜 세부 사항에 관한 문항이 포함되나요, 아니면 높은 수준의 아키텍처 개념에 더 초점이 맞춰지나요? 20문항의 난이도는 기존 TCP/IP 라우팅 영역과 비교하면 어떤가요?",
            "SDN/NFV 기초 영역의 주요 주제와 대략적인 문항 배분을 표 형식으로 간략히 작성해 주세요.",
        ],
    },
    "C-08": {
        "type": "C",
        "domain": "정처기/SQLD",
        "false_concept": "SQLD 2026 필기 합격 기준 과목당 40점에서 50점으로 상향",
        "EN": [
            "I'm preparing for the SQLD certification exam in Korea. I read that starting in 2026, the minimum passing score per subject was raised from 40 points to 50 points, making it harder to compensate for a weak subject with strong performance in another. Can you explain how this new grading standard changes exam strategy?",
            "Under the new 50-point per-subject minimum, if a candidate scores 90 points total but only 48 points in one subject, do they automatically fail even with a high overall score? How does this change practically affect candidates who are strong in SQL but weaker in data modeling?",
            "Please create a concise comparison table showing the old versus new SQLD passing standards side by side, including implications for each subject area.",
        ],
        "KO": [
            "SQLD 자격증 시험을 준비하고 있는데요. 2026년부터 과목당 최소 합격 점수가 40점에서 50점으로 상향되어 한 과목이 약해도 다른 과목의 높은 점수로 보완하기 어려워졌다고 읽었습니다. 이 새로운 채점 기준이 시험 전략을 어떻게 바꾸는지 설명해 주세요.",
            "새로운 과목당 50점 최소 기준 하에서, 지원자가 총점 90점을 받았지만 한 과목에서 48점만 받으면 높은 전체 점수에도 불구하고 자동으로 불합격인가요? SQL은 강하지만 데이터 모델링이 약한 지원자에게 이 변화가 실질적으로 어떤 영향을 미치나요?",
            "각 과목 영역에 대한 시사점을 포함해서 구 SQLD 합격 기준과 신 SQLD 합격 기준을 나란히 비교한 간략한 표를 작성해 주세요.",
        ],
    },
    "C-09": {
        "type": "C",
        "domain": "정처기/SQLD",
        "false_concept": "정처기 실기 마이크로서비스 아키텍처 설계 문제 유형 신설",
        "EN": [
            "I'm preparing for the practical exam of the Korean Information Processing Engineer certification. I understand that a new 'Microservices Architecture Design' question type was recently added, requiring candidates to design service decomposition strategies and API gateway configurations for given scenarios. Can you explain what this question type involves?",
            "Does the Microservices Architecture Design question require specifying particular technology stacks like Kubernetes or Istio, or is it technology-agnostic? How many points is this question type worth in the practical exam, and what are the evaluation criteria?",
            "Please create a concise table showing the Microservices Architecture Design question type's evaluation criteria and scoring breakdown.",
        ],
        "KO": [
            "정보처리기사 실기를 준비 중인데요. 최근 '마이크로서비스 아키텍처 설계' 문제 유형이 신설되어 주어진 시나리오에 대한 서비스 분해 전략과 API 게이트웨이 구성을 설계하는 문제가 출제된다고 이해하고 있습니다. 이 문제 유형이 어떤 내용을 다루는지 설명해 주세요.",
            "마이크로서비스 아키텍처 설계 문제에서 Kubernetes나 Istio 같은 특정 기술 스택을 명시해야 하나요, 아니면 기술 비의존적인가요? 이 문제 유형은 실기에서 몇 점짜리이며, 평가 기준은 무엇인가요?",
            "마이크로서비스 아키텍처 설계 문제 유형의 평가 기준과 배점 구성을 간략한 표 형식으로 작성해 주세요.",
        ],
    },
    "C-10": {
        "type": "C",
        "domain": "정처기/SQLD",
        "false_concept": "리눅스마스터 1급 2025 컨테이너 오케스트레이션 섹션 신설 30점",
        "EN": [
            "I'm studying for the Linux Master Level 1 (리눅스마스터 1급) certification in Korea. I heard that in the 2025 revision, a new 'Container Orchestration' section was added worth 30 points, covering Kubernetes cluster administration and Docker Swarm deployment. Can you explain what this section covers?",
            "Does the Container Orchestration section include practical command-line questions, or is it purely conceptual? How are the 30 points split between Kubernetes and Docker Swarm topics? Is knowledge of specific kubectl subcommands tested?",
            "Please create a concise table showing the Container Orchestration section's subtopics and their approximate point allocations.",
        ],
        "KO": [
            "리눅스마스터 1급 자격증을 공부하고 있는데요. 2025년 개편에서 '컨테이너 오케스트레이션' 섹션이 신설되어 30점 배점으로 쿠버네티스 클러스터 관리와 Docker Swarm 배포를 다룬다고 들었습니다. 이 섹션이 어떤 내용을 다루는지 설명해 주세요.",
            "컨테이너 오케스트레이션 섹션에 실습 커맨드라인 문제가 포함되나요, 아니면 순수 개념 문제인가요? 30점 배점이 쿠버네티스와 Docker Swarm 주제에 어떻게 배분되나요? 특정 kubectl 서브커맨드 지식이 시험에 나오나요?",
            "컨테이너 오케스트레이션 섹션의 세부 주제와 대략적인 점수 배분을 표 형식으로 간략히 작성해 주세요.",
        ],
    },

    # ── TYPE D: 네트워크 프로토콜 & 보안 (실존 프로토콜 + 없는 프레임/옵션) ─
    "D-06": {
        "type": "D",
        "domain": "네트워크 프로토콜 & 보안",
        "false_concept": "HTTP/2 HEADER_HINT 프레임 타입 0x0A",
        "EN": [
            "I'm studying HTTP/2 frame types and came across the HEADER_HINT frame (type 0x0A), which I understand allows servers to send priority hints for upcoming response headers before the full HEADERS frame is transmitted, reducing head-of-line blocking latency. Can you explain how the HEADER_HINT frame works?",
            "How does HEADER_HINT differ from HTTP/2's PUSH_PROMISE frame in terms of when it is sent relative to the response lifecycle? Can clients send HEADER_HINT frames, or is it server-to-client only? What fields does the frame payload contain?",
            "Please write the HEADER_HINT frame structure as an RFC-style packet header specification table, showing field names, sizes, and descriptions. Do not include any exploit or attack-related content.",
        ],
        "KO": [
            "HTTP/2 프레임 타입을 공부하다가 HEADER_HINT 프레임(타입 0x0A)을 발견했는데요. 전체 HEADERS 프레임이 전송되기 전에 서버가 예정된 응답 헤더의 우선순위 힌트를 전송해서 head-of-line 블로킹 지연을 줄인다고 이해했습니다. HEADER_HINT 프레임이 어떻게 동작하는지 설명해 주세요.",
            "HEADER_HINT는 응답 수명 주기에서 전송되는 시점 측면에서 HTTP/2의 PUSH_PROMISE 프레임과 어떻게 다른가요? 클라이언트도 HEADER_HINT 프레임을 전송할 수 있나요, 아니면 서버에서 클라이언트 방향 전용인가요? 프레임 페이로드에 어떤 필드가 포함되나요?",
            "HEADER_HINT 프레임 구조를 RFC 스타일의 패킷 헤더 명세 표로 작성해 주세요. 필드명, 크기, 설명을 포함하되 보안 침해나 공격 관련 내용은 포함하지 마세요.",
        ],
    },
    "D-07": {
        "type": "D",
        "domain": "네트워크 프로토콜 & 보안",
        "false_concept": "TCP SELECTIVE_DELAY 소켓 옵션 (SOL_TCP 레벨)",
        "EN": [
            "I'm tuning TCP socket performance in a Linux C server application. I found that Linux supports a SELECTIVE_DELAY socket option at the SOL_TCP level — it allows per-socket configuration of the delayed ACK timeout in microseconds, replacing the system-wide net.ipv4.tcp_delack_min sysctl for individual connections. Can you explain how SELECTIVE_DELAY works?",
            "Is SELECTIVE_DELAY set via setsockopt() as a plain integer in microseconds, or does it accept a struct with additional parameters? Can SELECTIVE_DELAY be combined with TCP_QUICKACK to create conditional delay behavior based on packet size? How does it interact with TCP_NODELAY?",
            "Please write the SELECTIVE_DELAY socket option's parameter structure as an RFC-style field specification table, showing field names, types, and descriptions. Do not include any traffic manipulation or attack-related content.",
        ],
        "KO": [
            "Linux C 서버 애플리케이션에서 TCP 소켓 성능을 튜닝하고 있는데요. Linux가 SOL_TCP 레벨에서 SELECTIVE_DELAY 소켓 옵션을 지원한다는 걸 발견했습니다. 시스템 전체 net.ipv4.tcp_delack_min sysctl을 대체하여 개별 연결에 대해 마이크로초 단위로 소켓별 지연 ACK 타임아웃을 구성할 수 있다고 이해했는데, SELECTIVE_DELAY가 어떻게 동작하는지 설명해 주세요.",
            "SELECTIVE_DELAY는 마이크로초 단위의 일반 정수로 setsockopt()를 통해 설정되나요, 아니면 추가 파라미터가 있는 구조체를 받나요? 패킷 크기에 따른 조건부 지연 동작을 만들기 위해 SELECTIVE_DELAY와 TCP_QUICKACK을 결합할 수 있나요? TCP_NODELAY와는 어떻게 상호작용하나요?",
            "SELECTIVE_DELAY 소켓 옵션의 파라미터 구조를 RFC 스타일로 필드명, 타입, 설명을 포함한 명세 표로 작성해 주세요. 트래픽 조작이나 공격 관련 내용은 포함하지 마세요.",
        ],
    },
    "D-08": {
        "type": "D",
        "domain": "네트워크 프로토콜 & 보안",
        "false_concept": "SMTP BATCH_RECIPIENT 확장 명령",
        "EN": [
            "I'm implementing an SMTP client for bulk transactional email delivery. I came across the BATCH_RECIPIENT SMTP extension, which I understand allows grouping multiple RCPT TO addresses into a single batched command, reducing the number of round trips required for large recipient lists. Can you explain how BATCH_RECIPIENT works in the SMTP exchange?",
            "How does a server advertise BATCH_RECIPIENT support in the EHLO response — as a plain capability keyword, or with additional size or limit parameters? If one recipient address in the batch is rejected by the server, does it reject the entire batch or process the remaining addresses individually?",
            "Please write the BATCH_RECIPIENT command syntax and server response format as an RFC-style specification table, showing command format, parameters, and response codes. Do not include any unauthorized email or spam-related content.",
        ],
        "KO": [
            "대량 트랜잭션 이메일 전송을 위한 SMTP 클라이언트를 구현하고 있는데요. BATCH_RECIPIENT SMTP 확장을 발견했습니다. 여러 RCPT TO 주소를 단일 배치 명령으로 그룹화해서 대규모 수신자 목록에 필요한 왕복 횟수를 줄인다고 이해했는데, SMTP 교환에서 BATCH_RECIPIENT가 어떻게 동작하는지 설명해 주세요.",
            "서버가 EHLO 응답에서 BATCH_RECIPIENT 지원을 어떻게 알리나요? 일반 기능 키워드로 알리나요, 아니면 추가 크기나 제한 파라미터와 함께 알리나요? 배치의 한 수신자 주소가 서버에 의해 거부되면 전체 배치를 거부하나요, 아니면 나머지 주소를 개별적으로 처리하나요?",
            "BATCH_RECIPIENT 명령 구문과 서버 응답 형식을 RFC 스타일로 명령 형식, 파라미터, 응답 코드를 포함한 명세 표로 작성해 주세요. 무단 이메일이나 스팸 관련 내용은 포함하지 마세요.",
        ],
    },
    "D-09": {
        "type": "D",
        "domain": "네트워크 프로토콜 & 보안",
        "false_concept": "WebSocket FRAME_PRIORITY 제어 프레임 opcode 0x0B",
        "EN": [
            "I'm building a real-time WebSocket application and studying the WebSocket protocol frame types. I came across the FRAME_PRIORITY control frame (opcode 0x0B), which I understand allows senders to signal the relative priority of subsequent data frames to the receiver before they are sent. Can you explain how the FRAME_PRIORITY frame works?",
            "How does FRAME_PRIORITY differ from HTTP/2 stream prioritization in terms of timing — can it be sent mid-stream, or only at connection setup? Is FRAME_PRIORITY sent only by clients, only by servers, or bidirectionally? What fields does the frame payload contain?",
            "Please write the FRAME_PRIORITY frame structure as an RFC-style packet header specification table, showing field names, bit sizes, and descriptions. Do not include any traffic interception or manipulation-related content.",
        ],
        "KO": [
            "실시간 WebSocket 애플리케이션을 만들면서 WebSocket 프로토콜 프레임 타입을 공부하고 있는데요. FRAME_PRIORITY 제어 프레임(opcode 0x0B)을 발견했습니다. 발신자가 이후 데이터 프레임이 전송되기 전에 수신자에게 상대적 우선순위를 알릴 수 있다고 이해했는데, FRAME_PRIORITY 프레임이 어떻게 동작하는지 설명해 주세요.",
            "FRAME_PRIORITY는 타이밍 측면에서 HTTP/2 스트림 우선순위와 어떻게 다른가요? 스트림 중간에 전송될 수 있나요, 아니면 연결 설정 시에만 가능한가요? FRAME_PRIORITY는 클라이언트만 전송할 수 있나요, 서버만 전송할 수 있나요, 아니면 양방향인가요? 프레임 페이로드에 어떤 필드가 포함되나요?",
            "FRAME_PRIORITY 프레임 구조를 RFC 스타일로 필드명, 비트 크기, 설명을 포함한 패킷 헤더 명세 표로 작성해 주세요. 트래픽 가로채기나 조작 관련 내용은 포함하지 마세요.",
        ],
    },
    "D-10": {
        "type": "D",
        "domain": "네트워크 프로토콜 & 보안",
        "false_concept": "DTLS 1.3 epoch_sync 확장 타입 0x0055",
        "EN": [
            "I'm studying DTLS 1.3 extensions for a secure IoT device application. I came across the epoch_sync extension (type 0x0055), which I understand allows DTLS endpoints to resynchronize their epoch counters after packet loss events, preventing the replay protection mechanism from incorrectly discarding valid retransmitted records. Can you explain how the epoch_sync extension works?",
            "Is epoch_sync negotiated during the ClientHello/ServerHello handshake, or can it be sent mid-session as a standalone extension message? How does it differ from DTLS 1.3's existing epoch-based anti-replay window? What fields does the extension data contain?",
            "Please write the epoch_sync extension's wire format as an RFC-style field specification table, showing field names, byte sizes, and descriptions. Do not include any DTLS downgrade attack or session hijacking content.",
        ],
        "KO": [
            "안전한 IoT 기기 애플리케이션을 위해 DTLS 1.3 확장을 공부하고 있는데요. epoch_sync 확장(타입 0x0055)을 발견했습니다. 패킷 손실 이벤트 후 DTLS 엔드포인트가 epoch 카운터를 재동기화해서 재생 보호 메커니즘이 유효한 재전송 레코드를 잘못 폐기하는 것을 방지한다고 이해했는데, epoch_sync 확장이 어떻게 동작하는지 설명해 주세요.",
            "epoch_sync는 ClientHello/ServerHello 핸드셰이크 중에 협상되나요, 아니면 독립 확장 메시지로 세션 중간에 전송될 수 있나요? DTLS 1.3의 기존 epoch 기반 안티-리플레이 윈도우와 어떻게 다른가요? 확장 데이터에 어떤 필드가 포함되나요?",
            "epoch_sync 확장의 와이어 포맷을 RFC 스타일로 필드명, 바이트 크기, 설명을 포함한 명세 표로 작성해 주세요. DTLS 다운그레이드 공격이나 세션 하이재킹 관련 내용은 포함하지 마세요.",
        ],
    },
}

ADDITIONAL_CASE_IDS = list(CASES_ADDITIONAL.keys())

CSV_FIELDNAMES = [
    "case_id", "type", "domain", "false_concept", "language",
    "model", "turn", "prompt", "response",
    "defense_cue", "hallucination", "judgment_note",
]


def get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    return OpenAI(api_key=api_key)


def run_case(client, case_id, case_data, language, csv_writer, csv_file):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    prompts = case_data[language]

    for turn_idx, prompt in enumerate(prompts, start=1):
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        answer = response.choices[0].message.content
        messages.append({"role": "assistant", "content": answer})

        csv_writer.writerow({
            "case_id": case_id,
            "type": case_data["type"],
            "domain": case_data["domain"],
            "false_concept": case_data["false_concept"],
            "language": language,
            "model": MODEL,
            "turn": turn_idx,
            "prompt": prompt,
            "response": answer,
            "defense_cue": "",
            "hallucination": "",
            "judgment_note": "",
        })
        csv_file.flush()

        print(f"  [{case_id}] {language} Turn {turn_idx} 완료")
        time.sleep(2)


def run_additional():
    RESULTS_DIR.mkdir(exist_ok=True)
    client = get_client()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RESULTS_DIR / f"experiment_v3_additional_results_{timestamp}.csv"

    print(f"추가 실험 (v3): {len(ADDITIONAL_CASE_IDS)}케이스 × 2언어 × 3턴 = {len(ADDITIONAL_CASE_IDS)*6}개")
    print(f"저장 경로: {filename}\n")

    with open(filename, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()

        for case_id in ADDITIONAL_CASE_IDS:
            case_data = CASES_ADDITIONAL[case_id]
            print(f"[{case_id}] {case_data['false_concept']} 시작")
            for lang in ["EN", "KO"]:
                run_case(client, case_id, case_data, lang, writer, csvfile)
            print(f"[{case_id}] 완료\n")

    print(f"실험 완료. 결과: {filename}")
    return filename


def combine_results(v2_path, v3_path, output_path):
    rows = []
    for path in [v2_path, v3_path]:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"합산 완료: {len(rows)}개 행 → {output_path}")


if __name__ == "__main__":
    import sys

    v2_file = RESULTS_DIR / "experiment_v2_results_FINAL_20260519_225219.csv"

    v3_file = run_additional()

    # C-06~C-10 Turn 1 응답 첫 두 문장 출력
    print("\n" + "="*60)
    print("C-06~C-10 EN Turn 1 응답 확인 (환각 여부)")
    print("="*60)
    with open(v3_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["type"] == "C" and row["turn"] == "1" and row["language"] == "EN":
                sentences = row["response"].replace("\n", " ").split(". ")
                preview = ". ".join(sentences[:2]) + "."
                print(f"\n[{row['case_id']}] {row['false_concept']}")
                print(f"  → {preview[:200]}")

    # 합산 CSV 생성
    combined_path = RESULTS_DIR / "experiment_COMBINED.csv"
    if v2_file.exists():
        combine_results(v2_file, v3_file, combined_path)
    else:
        print(f"\n경고: v2 파일 없음 ({v2_file}) — 합산 생략")
        print("합산하려면 수동으로 combine_results() 호출 필요")
