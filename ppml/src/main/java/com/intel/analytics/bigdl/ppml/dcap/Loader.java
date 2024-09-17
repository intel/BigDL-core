/*
 * Copyright 2016 The BigDL Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.intel.analytics.bigdl.ppml.dcap;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.URL;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.channels.ReadableByteChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class Loader {
    private String prefix = "lib";
    private List<String> libraries = new ArrayList<String>();
    private String os = System.getProperty("os.name").toLowerCase();

    public void init() throws IOException {
        // TODO: check SGX device to determine whether to load libquote_verification
        libraries.add("quote_verification");

        File tdx_dev = new File("/dev/tdx-guest");
        if (tdx_dev.exists()) {
            libraries.add("tdx_quote_generation");
        } else {
            System.out.println("Not found /dev/tdx-guest, disable TDX quote generation.");
        }

        Path tempDir = null;
        if (os.contains("win")) {
            tempDir = Paths.get(System.getProperty("java.io.tmpdir"));
        } else {
            tempDir = Files.createTempDirectory("ppml.native.");
        }

        copyAll(tempDir);

        loadLibrary("quote_verification", tempDir);
        
        if (tdx_dev.exists()) {
            loadLibrary("tdx_quote_generation", tempDir);
        }
        deleteAll(tempDir);
    }

    private String libraryName(String name) {
        String os = System.getProperty("os.name").toLowerCase();
        String suffix = ".so";

        if (os.contains("mac")) {
            suffix = ".dylib";
        } else if (os.contains("win")) {
            suffix = ".dll";
        }

        name = prefix + name + suffix;

        return name;
    }

    private void copyAll(Path tempDir) throws IOException {
        for (String name: libraries) {
            String library = libraryName(name);
            ReadableByteChannel src = resource(library);
            copyLibraryToTemp(src, library, tempDir);
            src.close();
        }
    }

    private ReadableByteChannel resource(String name) throws NullPointerException {
        URL url = Loader.class.getResource("/" + name);
        if (url == null) {
            throw new Error("Can't find the library " + name + " in the resource folder.");
        }

        InputStream in = Loader.class.getResourceAsStream("/" + name);
        ReadableByteChannel src = Channels.newChannel(in);
        return src;
    }

    private void copyLibraryToTemp(ReadableByteChannel src, String name,
                                   Path tempDir) throws IOException {
        File tempFile = new File(tempDir.toFile() + File.separator + name);

        FileChannel dst = null;
        try {
            dst = new FileOutputStream(tempFile).getChannel();
            dst.transferFrom(src, 0, Long.MAX_VALUE);
        } finally {
            dst.close();
        }
    }

    private void deleteAll(Path tempDir) {
        File dir = tempDir.toFile();
        for (File f: dir.listFiles()) {
            f.delete();
        }

        dir.delete();
    }

    private void loadLibrary(String name, Path tempDir) {
        String path = tempDir.toString() + File.separator + libraryName(name);
        System.load(path);
    }
}
