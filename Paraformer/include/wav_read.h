#ifndef WAV_READ_H_
#define WAV_READ_H_
#include <stdint.h>
#include <vector>
#include <fstream>

//Wav Header
struct wav_header_t
{
    int32_t chunkID; //"RIFF" = 0x46464952
    int32_t chunkSize; //28 [+ sizeof(wExtraFormatBytes) + wExtraFormatBytes] + sum(sizeof(chunk.id) + sizeof(chunk.size) + chunk.size)
    int32_t format; //"WAVE" = 0x45564157
    int32_t subchunk1ID; //"fmt " = 0x20746D66
    int32_t subchunk1Size; //16 [+ sizeof(wExtraFormatBytes) + wExtraFormatBytes]
    int16_t audioFormat;
    int16_t numChannels;
    int32_t sampleRate;
    int32_t byteRate;
    int16_t blockAlign;
    int16_t bitsPerSample;
};

//Chunks
struct chunk_t
{
    int32_t ID;//"data" = 0x61746164
    int32_t size;
};

int load_wav(const char* file_name, std::vector<float>& data){
    std::ifstream fin(file_name, std::ifstream::binary);
    if (!fin.is_open()) {
        fprintf(stderr, "can not open %s \n", file_name);
        return -1;
    }

    wav_header_t header;
    fin.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (header.chunkID != 0x46464952) {
        fprintf(stderr, "check chunkID RIFF failed: 0x%08x\n", header.chunkID);
        return -1;
    }

    if (header.format != 0x45564157) {
        fprintf(stderr, "check audio format failed: 0x%08x\n", header.format);
        return -1;
    }

    if (header.sampleRate != 16000) {
        fprintf(stderr, "only support 16000 for sample rate: %d\n", header.sampleRate);
        return -1;
    }

    chunk_t chunk;
    while (true){
        fin.read(reinterpret_cast<char*>(&chunk), sizeof(chunk));
        if (chunk.ID == 0x61746164)
            break;
        fin.seekg(chunk.size, std::istream::cur);
    }

    int sample_size = header.bitsPerSample / 8;
    int samples_count = chunk.size * 8 / header.bitsPerSample;
    
    int16_t* value = new int16_t[samples_count];
    std::memset(value, 0, sizeof(int16_t) * samples_count);
    fin.read(reinterpret_cast<char*>(value), chunk.size);

    data.resize(samples_count);
    for (int i = 0; i < samples_count; ++i) {
        data[i] = static_cast<float>(value[i * header.numChannels]) / 32768.;
    }

    fin.close();
    delete [] value;

    return 0;
}

#endif
