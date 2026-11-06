package gateway

import (
	"encoding/json"
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/yetone/magpie/internal/provider"
)

// isMistral is the provider's vendor: the preset, a provider added by its
// address, and a plugin in front of it (Mistral Vibe Code's, whose own host
// is plugin://<id>).
func TestIsMistral(t *testing.T) {
	if !isMistral(provider.Provider{Preset: "mistral"}) {
		t.Error("the mistral preset is not Mistral")
	}
	if !isMistral(provider.Provider{Chat: "https://api.mistral.ai/v1"}) {
		t.Error("a provider added at api.mistral.ai is not Mistral")
	}
	if !isMistral(provider.PluginForTest("mistral-vibe", "mistral-vibe", "https://api.mistral.ai/v1")) {
		t.Error("a plugin in front of api.mistral.ai is not Mistral")
	}
	if isMistral(provider.PluginForTest("sensenova", "sensenova", "https://token.sensenova.cn/v1")) {
		t.Error("another plugin's vendor is Mistral")
	}
	if isMistral(provider.Provider{Chat: "https://api.openai.com/v1"}) {
		t.Error("OpenAI is Mistral")
	}
}

// Claude Code speaks Anthropic's Messages API and Mistral serves chat
// completions alone, so its request is translated: an earlier turn's
// thinking, which Claude sends in a thinking block, the build wrote as
// reasoning_content, which Mistral turns the whole request away for (422
// extra_forbidden, loc ["body","messages",1,"assistant","reasoning_content"]).
// It goes as the typed thinking part Mistral returns it in, and the strict
// upstream takes the request at once (#494).
func TestMistralThinkingTranslatedFromAnthropic(t *testing.T) {
	// Mistral's message models are the strict ones: a field it doesn't know
	// there is a 422 extra_forbidden located in the messages. The top-level
	// fields are mistral_extra_fields' side (the build adds stream_options to
	// every streamed request), so only the messages' verdict is taken here.
	refuse := func(body []byte) (int, string) {
		var q struct {
			Messages []map[string]json.RawMessage `json:"messages"`
		}
		json.Unmarshal(body, &q)
		known := []string{"role", "content", "tool_calls", "tool_call_id", "name", "prefix"}
		var detail []string
		for i, m := range q.Messages {
			var role string
			json.Unmarshal(m["role"], &role)
			for k, v := range m {
				if slices.Contains(known, k) {
					continue
				}
				detail = append(detail, fmt.Sprintf(`{"type":"extra_forbidden","loc":["body","messages",%d,%q,%q],"msg":"Extra inputs are not permitted","input":%s}`, i, role, k, v))
			}
		}
		if len(detail) == 0 {
			return 0, ""
		}
		return 422, `{"detail":[` + strings.Join(detail, ",") + `]}`
	}
	// Claude asked for one JSON body; the Chat upstream is streamed and the
	// reply gathered for it
	f := &fake{t: t, reply: sse(
		`data: {"id":"c1","model":"m1","choices":[{"delta":{"content":"No."}}]}`,
		`data: {"id":"c1","choices":[{"delta":{},"finish_reason":"stop"}]}`,
		`data: [DONE]`), refuse: refuse}
	up := setup(t, provider.Chat, f)
	// a model Mistral serves that wants its thinking back: the build replays
	// it as reasoning_content wherever the model is served
	if err := provider.Save(provider.Provider{ID: "fake", Name: "Fake", Key: "k", Models: []string{"deepseek-v4-1-flash-1017"},
		Preset: "mistral", Chat: up.URL + "/v1"}); err != nil {
		t.Fatal(err)
	}
	srv := New()
	code, body := postTo(t, srv, "/v1/messages", `{"model":"deepseek-v4-1-flash-1017","max_tokens":64,"messages":[
        {"role":"user","content":"Is 91 prime?"},
        {"role":"assistant","content":[{"type":"thinking","thinking":"7*13=91.","signature":""},{"type":"text","text":"91 = 7 x 13."}]},
        {"role":"user","content":"And 97?"}]}`)
	if code != 200 || f.calls != 1 {
		t.Fatalf("%d after %d calls: %s", code, f.calls, body)
	}
	if strings.Contains(string(f.got), "reasoning_content") {
		t.Errorf("reasoning_content sent to Mistral: %s", f.got)
	}
	ms := sentMessages(t, f.got)
	sameJSON(t, ms[1], `{"role":"assistant","content":[{"type":"thinking","thinking":[{"type":"text","text":"7*13=91."}]},{"type":"text","text":"91 = 7 x 13."}]}`)
	sameJSON(t, ms[0], `{"role":"user","content":"Is 91 prime?"}`)
	sameJSON(t, ms[2], `{"role":"user","content":"And 97?"}`)
}
